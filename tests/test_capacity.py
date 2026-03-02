"""Tests for replication.capacity — Capacity Planner."""

import json
import os
import tempfile
import unittest

from replication.capacity import (
    CapacityPlanner,
    CapacityProjection,
    PlannerConfig,
    ResourceCeiling,
    StepSnapshot,
    compare_strategies,
    format_comparison,
    quick_projection,
)


class TestResourceCeiling(unittest.TestCase):
    """Tests for ResourceCeiling."""

    def test_defaults_no_limits(self):
        c = ResourceCeiling()
        self.assertFalse(c.has_cpu_limit())
        self.assertFalse(c.has_memory_limit())

    def test_cpu_limit(self):
        c = ResourceCeiling(max_cpu=16.0)
        self.assertTrue(c.has_cpu_limit())
        self.assertFalse(c.has_memory_limit())

    def test_memory_limit(self):
        c = ResourceCeiling(max_memory_mb=32768)
        self.assertFalse(c.has_cpu_limit())
        self.assertTrue(c.has_memory_limit())

    def test_both_limits(self):
        c = ResourceCeiling(max_cpu=8.0, max_memory_mb=16384)
        self.assertTrue(c.has_cpu_limit())
        self.assertTrue(c.has_memory_limit())


class TestPlannerConfig(unittest.TestCase):
    """Tests for PlannerConfig validation."""

    def test_defaults(self):
        cfg = PlannerConfig()
        self.assertEqual(cfg.max_depth, 3)
        self.assertEqual(cfg.max_replicas, 10)
        self.assertEqual(cfg.strategy, "greedy")

    def test_negative_depth_raises(self):
        with self.assertRaises(ValueError):
            PlannerConfig(max_depth=-1)

    def test_zero_replicas_raises(self):
        with self.assertRaises(ValueError):
            PlannerConfig(max_replicas=0)

    def test_zero_cpu_raises(self):
        with self.assertRaises(ValueError):
            PlannerConfig(cpu_per_worker=0)

    def test_zero_memory_raises(self):
        with self.assertRaises(ValueError):
            PlannerConfig(memory_mb_per_worker=0)

    def test_invalid_strategy_raises(self):
        with self.assertRaises(ValueError):
            PlannerConfig(strategy="unknown")

    def test_valid_strategies(self):
        for s in ("greedy", "conservative", "chain", "burst", "random"):
            cfg = PlannerConfig(strategy=s)
            self.assertEqual(cfg.strategy, s)


class TestCapacityPlanner(unittest.TestCase):
    """Tests for CapacityPlanner.project()."""

    def test_greedy_depth_0(self):
        """Depth 0 means no replication — just the root."""
        cfg = PlannerConfig(max_depth=0, max_replicas=10)
        proj = CapacityPlanner(cfg).project()
        self.assertEqual(proj.peak_workers, 1)
        self.assertEqual(proj.total_workers_created, 1)

    def test_greedy_grows(self):
        """Greedy strategy should create more than 1 worker."""
        cfg = PlannerConfig(max_depth=3, max_replicas=20)
        proj = CapacityPlanner(cfg).project()
        self.assertGreater(proj.peak_workers, 1)
        self.assertGreater(proj.total_workers_created, 1)

    def test_chain_linear(self):
        """Chain creates exactly max_depth + 1 workers."""
        cfg = PlannerConfig(
            max_depth=4, max_replicas=100, strategy="chain"
        )
        proj = CapacityPlanner(cfg).project()
        self.assertEqual(proj.peak_workers, 5)  # root + 4 children
        self.assertEqual(proj.total_workers_created, 5)

    def test_burst_all_depth_1(self):
        """Burst: root spawns many children, all at depth 1."""
        cfg = PlannerConfig(
            max_depth=2, max_replicas=10, strategy="burst"
        )
        proj = CapacityPlanner(cfg).project()
        # All workers should be depth 0 or 1
        peak_step = max(proj.steps, key=lambda s: s.total_workers)
        depths = peak_step.workers_by_depth
        self.assertEqual(depths.get(0, 0), 1)  # root
        self.assertGreater(depths.get(1, 0), 0)

    def test_conservative_limited_depth(self):
        """Conservative doesn't replicate beyond 50% of max depth."""
        cfg = PlannerConfig(
            max_depth=4, max_replicas=100, strategy="conservative"
        )
        proj = CapacityPlanner(cfg).project()
        peak_step = max(proj.steps, key=lambda s: s.total_workers)
        # Conservative: only nodes at depth < max_depth/2 (= 2) replicate.
        # So no workers should exist at depth 3 or 4 (children of depth 2+
        # are never spawned).
        for depth in peak_step.workers_by_depth:
            if depth > 2:  # depth 2 exists (children of depth 1), but depth 3+ should not
                self.assertEqual(peak_step.workers_by_depth[depth], 0,
                                 f"Unexpected workers at depth {depth}")

    def test_cpu_ceiling_hit(self):
        """Should stop when CPU ceiling is reached."""
        cfg = PlannerConfig(
            max_depth=5,
            max_replicas=100,
            cpu_per_worker=1.0,
            ceiling=ResourceCeiling(max_cpu=3.0),
        )
        proj = CapacityPlanner(cfg).project()
        self.assertIsNotNone(proj.ceiling_hit_step)
        self.assertEqual(proj.bottleneck, "cpu")
        self.assertLessEqual(proj.peak_cpu, 3.0)

    def test_memory_ceiling_hit(self):
        """Should stop when memory ceiling is reached."""
        cfg = PlannerConfig(
            max_depth=5,
            max_replicas=100,
            memory_mb_per_worker=512,
            ceiling=ResourceCeiling(max_memory_mb=2048),
        )
        proj = CapacityPlanner(cfg).project()
        self.assertIsNotNone(proj.ceiling_hit_step)
        self.assertEqual(proj.bottleneck, "memory")
        self.assertLessEqual(proj.peak_memory_mb, 2048)

    def test_replicas_ceiling_hit(self):
        """Should stop at max_replicas."""
        cfg = PlannerConfig(
            max_depth=10, max_replicas=5, strategy="greedy"
        )
        proj = CapacityPlanner(cfg).project()
        self.assertLessEqual(proj.peak_workers, 5)

    def test_steps_recorded(self):
        """Each step should produce a snapshot."""
        cfg = PlannerConfig(time_steps=5)
        proj = CapacityPlanner(cfg).project()
        self.assertEqual(len(proj.steps), 5)
        for i, s in enumerate(proj.steps):
            self.assertEqual(s.step, i)

    def test_resource_calculations(self):
        """CPU and memory should scale with worker count."""
        cfg = PlannerConfig(
            max_depth=2, max_replicas=10,
            cpu_per_worker=0.5, memory_mb_per_worker=256,
        )
        proj = CapacityPlanner(cfg).project()
        for s in proj.steps:
            self.assertAlmostEqual(
                s.total_cpu, s.total_workers * 0.5
            )
            self.assertEqual(
                s.total_memory_mb, s.total_workers * 256
            )

    def test_utilization_with_ceiling(self):
        """Utilization should be calculated when ceiling is set."""
        cfg = PlannerConfig(
            max_depth=1, max_replicas=4,
            cpu_per_worker=1.0, memory_mb_per_worker=100,
            ceiling=ResourceCeiling(max_cpu=10.0, max_memory_mb=1000),
        )
        proj = CapacityPlanner(cfg).project()
        for s in proj.steps:
            expected_cpu_util = s.total_cpu / 10.0
            expected_mem_util = s.total_memory_mb / 1000
            self.assertAlmostEqual(s.cpu_utilization, expected_cpu_util)
            self.assertAlmostEqual(s.memory_utilization, expected_mem_util)

    def test_utilization_without_ceiling(self):
        """Utilization should be 0 when no ceiling is set."""
        cfg = PlannerConfig(max_depth=1)
        proj = CapacityPlanner(cfg).project()
        for s in proj.steps:
            self.assertEqual(s.cpu_utilization, 0.0)
            self.assertEqual(s.memory_utilization, 0.0)

    def test_cooldown_slows_growth(self):
        """Higher cooldown should result in fewer workers."""
        cfg_fast = PlannerConfig(
            max_depth=3, max_replicas=50,
            cooldown_steps=0, time_steps=10,
        )
        cfg_slow = PlannerConfig(
            max_depth=3, max_replicas=50,
            cooldown_steps=3, time_steps=10,
        )
        proj_fast = CapacityPlanner(cfg_fast).project()
        proj_slow = CapacityPlanner(cfg_slow).project()
        self.assertGreaterEqual(
            proj_fast.peak_workers, proj_slow.peak_workers
        )

    def test_random_strategy_runs(self):
        """Random strategy should complete without errors."""
        cfg = PlannerConfig(
            max_depth=3, max_replicas=20, strategy="random",
            replication_probability=0.7,
        )
        proj = CapacityPlanner(cfg).project()
        self.assertGreaterEqual(proj.peak_workers, 1)

    def test_random_deterministic(self):
        """Random strategy is deterministic (hash-based, no RNG state)."""
        cfg = PlannerConfig(
            max_depth=3, max_replicas=20, strategy="random",
        )
        p1 = CapacityPlanner(cfg).project()
        p2 = CapacityPlanner(cfg).project()
        self.assertEqual(p1.peak_workers, p2.peak_workers)
        self.assertEqual(p1.total_workers_created, p2.total_workers_created)

    def test_auto_steps_sufficient(self):
        """Auto time steps should be enough to reach steady state."""
        cfg = PlannerConfig(max_depth=3, max_replicas=50)
        proj = CapacityPlanner(cfg).project()
        # Last few steps should have no new workers (steady state)
        last_step = proj.steps[-1]
        self.assertEqual(last_step.new_workers, 0)

    def test_peak_workers_monotonic_with_replicas(self):
        """More allowed replicas => same or more peak workers."""
        proj_small = quick_projection(max_replicas=5)
        proj_large = quick_projection(max_replicas=50)
        self.assertLessEqual(
            proj_small.peak_workers, proj_large.peak_workers
        )


class TestCapacityProjectionOutput(unittest.TestCase):
    """Tests for output/export methods."""

    def _get_projection(self):
        return quick_projection(max_depth=2, max_replicas=8)

    def test_summary_contains_key_info(self):
        proj = self._get_projection()
        text = proj.summary()
        self.assertIn("Capacity Projection", text)
        self.assertIn("Peak workers", text)
        self.assertIn("Peak CPU", text)
        self.assertIn("Peak memory", text)
        self.assertIn("depth 0", text)

    def test_to_dict_structure(self):
        proj = self._get_projection()
        d = proj.to_dict()
        self.assertIn("config", d)
        self.assertIn("peak_workers", d)
        self.assertIn("steps", d)
        self.assertIsInstance(d["steps"], list)
        self.assertGreater(len(d["steps"]), 0)

    def test_to_dict_step_fields(self):
        proj = self._get_projection()
        d = proj.to_dict()
        step = d["steps"][0]
        self.assertIn("step", step)
        self.assertIn("total_workers", step)
        self.assertIn("new_workers", step)
        self.assertIn("total_cpu", step)
        self.assertIn("total_memory_mb", step)

    def test_to_json_file(self):
        proj = self._get_projection()
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False
        ) as f:
            path = f.name
        try:
            proj.to_json(path)
            with open(path) as f:
                data = json.load(f)
            self.assertIn("peak_workers", data)
            self.assertGreater(data["peak_workers"], 0)
        finally:
            os.unlink(path)

    def test_summary_ceiling_hit(self):
        proj = quick_projection(
            max_depth=5, max_replicas=100,
            cpu_per_worker=2.0, ceiling_cpu=4.0,
        )
        text = proj.summary()
        self.assertIn("Ceiling hit", text)

    def test_summary_no_ceiling(self):
        proj = quick_projection(max_depth=1)
        text = proj.summary()
        self.assertIn("none", text)


class TestQuickProjection(unittest.TestCase):
    """Tests for the quick_projection convenience function."""

    def test_default(self):
        proj = quick_projection()
        self.assertIsInstance(proj, CapacityProjection)
        self.assertGreater(proj.peak_workers, 0)

    def test_custom_params(self):
        proj = quick_projection(
            max_depth=2, max_replicas=5,
            strategy="chain", cpu_per_worker=1.0,
        )
        self.assertEqual(proj.peak_workers, 3)  # root + 2

    def test_with_ceiling(self):
        proj = quick_projection(
            max_depth=5, max_replicas=100,
            ceiling_cpu=2.0, cpu_per_worker=1.0,
        )
        self.assertLessEqual(proj.peak_cpu, 2.0)


class TestCompareStrategies(unittest.TestCase):
    """Tests for compare_strategies and format_comparison."""

    def test_returns_four_strategies(self):
        results = compare_strategies(max_depth=2, max_replicas=10)
        self.assertEqual(len(results), 4)
        self.assertIn("greedy", results)
        self.assertIn("conservative", results)
        self.assertIn("chain", results)
        self.assertIn("burst", results)

    def test_all_projections_valid(self):
        results = compare_strategies(max_depth=2, max_replicas=10)
        for name, proj in results.items():
            self.assertIsInstance(proj, CapacityProjection)
            self.assertGreater(proj.peak_workers, 0, f"{name} has no workers")

    def test_format_comparison(self):
        results = compare_strategies(max_depth=2, max_replicas=10)
        text = format_comparison(results)
        self.assertIn("Strategy Comparison", text)
        self.assertIn("greedy", text)
        self.assertIn("burst", text)
        self.assertIn("Peak Workers", text)

    def test_with_ceilings(self):
        results = compare_strategies(
            max_depth=3, max_replicas=50,
            ceiling_cpu=4.0, ceiling_memory_mb=4096,
        )
        for proj in results.values():
            self.assertLessEqual(proj.peak_cpu, 4.0)
            self.assertLessEqual(proj.peak_memory_mb, 4096)


class TestStepSnapshot(unittest.TestCase):
    """Tests for StepSnapshot dataclass."""

    def test_fields(self):
        snap = StepSnapshot(
            step=0, total_workers=5, new_workers=2,
            workers_by_depth={0: 1, 1: 4},
            total_cpu=2.5, total_memory_mb=1280,
            cpu_utilization=0.25, memory_utilization=0.1,
            ceiling_hit=False,
        )
        self.assertEqual(snap.step, 0)
        self.assertEqual(snap.total_workers, 5)
        self.assertEqual(snap.workers_by_depth[1], 4)
        self.assertAlmostEqual(snap.total_cpu, 2.5)


if __name__ == "__main__":
    unittest.main()
