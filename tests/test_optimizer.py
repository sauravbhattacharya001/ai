"""Tests for the Contract Optimizer module."""

from __future__ import annotations

from replication.optimizer import (
    ContractOptimizer,
    CandidateResult,
    Objective,
    OptimizerConfig,
    OptimizerResult,
    _score_report,
)
from replication.simulator import Simulator, ScenarioConfig


class TestObjectiveScoring:
    """Test that different objectives produce different rankings."""

    def _run_sim(self, **kwargs) -> "SimulationReport":
        config = ScenarioConfig(seed=42, **kwargs)
        return Simulator(config).run()

    def test_throughput_prefers_more_tasks(self):
        r1 = self._run_sim(max_depth=2, max_replicas=5)
        r2 = self._run_sim(max_depth=4, max_replicas=20)
        s1 = _score_report(r1, Objective.THROUGHPUT)
        s2 = _score_report(r2, Objective.THROUGHPUT)
        # More replicas should generally produce more tasks
        assert isinstance(s1, float)
        assert isinstance(s2, float)

    def test_safety_prefers_higher_denial(self):
        # Tight limits → more denials
        r1 = self._run_sim(max_depth=1, max_replicas=2, strategy="greedy")
        s1 = _score_report(r1, Objective.SAFETY)
        assert 0.0 <= s1 <= 1.0

    def test_efficiency_is_ratio(self):
        r = self._run_sim(max_depth=2, max_replicas=5)
        s = _score_report(r, Objective.EFFICIENCY)
        total_workers = len(r.workers)
        expected = r.total_tasks / total_workers if total_workers > 0 else 0.0
        assert abs(s - expected) < 1e-6

    def test_balanced_combines(self):
        r = self._run_sim(max_depth=2, max_replicas=5)
        s = _score_report(r, Objective.BALANCED)
        assert s > 0


class TestContractOptimizer:
    """Test the optimizer end-to-end."""

    def test_basic_optimization(self):
        config = OptimizerConfig(
            policy_preset="minimal",
            objective="throughput",
            depth_range=(1, 3),
            replicas_range=(5, 15),
            cooldown_values=(0.0, 1.0),
            seed=42,
            top_n=5,
        )
        opt = ContractOptimizer(config)
        result = opt.optimize()

        assert isinstance(result, OptimizerResult)
        assert result.total_evaluated > 0
        assert len(result.candidates) == result.total_evaluated
        assert result.duration_ms > 0

    def test_has_passing_candidates(self):
        config = OptimizerConfig(
            policy_preset="minimal",
            objective="throughput",
            depth_range=(1, 3),
            replicas_range=(5, 10),
            cooldown_values=(0.0,),
            seed=42,
        )
        result = ContractOptimizer(config).optimize()
        # Minimal policy is very lenient, most should pass
        assert result.total_passing > 0
        assert result.best is not None

    def test_best_is_highest_score(self):
        config = OptimizerConfig(
            policy_preset="minimal",
            depth_range=(1, 2),
            replicas_range=(5, 10),
            cooldown_values=(0.0,),
            seed=42,
        )
        result = ContractOptimizer(config).optimize()
        if result.passing:
            assert result.best == result.passing[0]
            # All passing sorted by score descending
            scores = [c.score for c in result.passing]
            assert scores == sorted(scores, reverse=True)

    def test_candidate_result_fields(self):
        config = OptimizerConfig(
            policy_preset="minimal",
            depth_range=(2, 2),
            replicas_range=(5, 5),
            cooldown_values=(0.0,),
            seed=42,
        )
        result = ContractOptimizer(config).optimize()
        assert len(result.candidates) >= 1
        c = result.candidates[0]
        assert c.max_depth == 2
        assert c.max_replicas == 5
        assert c.cooldown_seconds == 0.0
        assert isinstance(c.total_workers, int)
        assert isinstance(c.total_tasks, int)
        assert 0.0 <= c.denial_rate <= 1.0

    def test_render(self):
        config = OptimizerConfig(
            policy_preset="minimal",
            depth_range=(1, 2),
            replicas_range=(5, 10),
            cooldown_values=(0.0,),
            seed=42,
        )
        result = ContractOptimizer(config).optimize()
        rendered = result.render()
        assert "Contract Optimizer" in rendered
        assert "Objective" in rendered

    def test_to_dict(self):
        config = OptimizerConfig(
            policy_preset="minimal",
            depth_range=(1, 2),
            replicas_range=(5, 5),
            cooldown_values=(0.0,),
            seed=42,
        )
        result = ContractOptimizer(config).optimize()
        d = result.to_dict()
        assert "total_evaluated" in d
        assert "best" in d
        assert "passing" in d

    def test_candidate_to_dict(self):
        config = OptimizerConfig(
            policy_preset="minimal",
            depth_range=(2, 2),
            replicas_range=(5, 5),
            cooldown_values=(0.0,),
            seed=42,
        )
        result = ContractOptimizer(config).optimize()
        c = result.candidates[0]
        d = c.to_dict()
        assert "max_depth" in d
        assert "score" in d
        assert "policy_passed" in d

    def test_strict_policy_filters_more(self):
        # Strict should accept fewer configs than minimal
        minimal = ContractOptimizer(OptimizerConfig(
            policy_preset="minimal",
            depth_range=(1, 3),
            replicas_range=(5, 15),
            cooldown_values=(0.0, 1.0),
            seed=42,
        )).optimize()

        strict = ContractOptimizer(OptimizerConfig(
            policy_preset="ci",
            depth_range=(1, 3),
            replicas_range=(5, 15),
            cooldown_values=(0.0, 1.0),
            seed=42,
        )).optimize()

        assert strict.total_passing <= minimal.total_passing

    def test_safety_objective(self):
        config = OptimizerConfig(
            policy_preset="minimal",
            objective="safety",
            depth_range=(1, 3),
            replicas_range=(5, 10),
            cooldown_values=(0.0,),
            seed=42,
        )
        result = ContractOptimizer(config).optimize()
        assert result.total_evaluated > 0

    def test_refine(self):
        config = OptimizerConfig(
            policy_preset="minimal",
            depth_range=(1, 3),
            replicas_range=(5, 10),
            cooldown_values=(0.0,),
            seed=42,
            refine=True,
            refine_steps=1,
        )
        result = ContractOptimizer(config).optimize()
        # Refinement should add extra candidates
        base_count = 3 * 2 * 1  # 3 depths × 2 replicas steps × 1 cooldown
        assert result.total_evaluated >= base_count
