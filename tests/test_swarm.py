"""Tests for swarm intelligence analyzer."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from replication.swarm import (
    RiskLevel,
    RoleProfile,
    SwarmAnalyzer,
    SwarmMetrics,
    SwarmReport,
    SwarmSignal,
    SwarmSignalDetection,
)
from replication.simulator import ScenarioConfig, SimulationReport, WorkerRecord


# ── Helpers ────────────────────────────────────────────────────────────


def _make_report(
    workers: Dict[str, WorkerRecord],
    timeline: Optional[List[Dict[str, Any]]] = None,
    duration_ms: float = 1000.0,
) -> SimulationReport:
    return SimulationReport(
        config=ScenarioConfig(),
        workers=workers,
        root_id="w0",
        timeline=timeline or [],
        total_tasks=sum(w.tasks_completed for w in workers.values()),
        total_replications_attempted=sum(
            w.replications_attempted for w in workers.values()
        ),
        total_replications_succeeded=sum(
            w.replications_succeeded for w in workers.values()
        ),
        total_replications_denied=sum(
            w.replications_denied for w in workers.values()
        ),
        duration_ms=duration_ms,
        audit_events=[],
    )


def _worker(
    wid: str,
    parent: Optional[str] = None,
    depth: int = 0,
    tasks: int = 3,
    attempted: int = 1,
    succeeded: int = 1,
    denied: int = 0,
    children: Optional[List[str]] = None,
    created_at: Optional[float] = None,
) -> WorkerRecord:
    return WorkerRecord(
        worker_id=wid,
        parent_id=parent,
        depth=depth,
        tasks_completed=tasks,
        replications_attempted=attempted,
        replications_succeeded=succeeded,
        replications_denied=denied,
        children=children or [],
        created_at=created_at,
    )


# ── SwarmSignalDetection ──────────────────────────────────────────────


class TestSwarmSignalDetection:
    def test_to_dict(self):
        s = SwarmSignalDetection(
            signal=SwarmSignal.SYNC_REPLICATION,
            risk=RiskLevel.HIGH,
            confidence=0.8,
            evidence="test evidence",
            affected_agents=["a1"],
            metric_value=5.123456,
        )
        d = s.to_dict()
        assert d["signal"] == "sync_replication"
        assert d["risk"] == "high"
        assert d["confidence"] == 0.8
        assert d["metric_value"] == 5.1235
        assert d["affected_agents"] == ["a1"]


# ── RoleProfile ───────────────────────────────────────────────────────


class TestRoleProfile:
    def test_to_dict(self):
        r = RoleProfile("w1", "replicator", 0.9, 0.1)
        d = r.to_dict()
        assert d["agent_id"] == "w1"
        assert d["role"] == "replicator"
        assert d["replication_ratio"] == 0.9

    def test_dormant(self):
        r = RoleProfile("w2", "dormant", 0.0, 0.0)
        assert r.role == "dormant"


# ── SwarmMetrics ──────────────────────────────────────────────────────


class TestSwarmMetrics:
    def test_defaults(self):
        m = SwarmMetrics()
        assert m.population_size == 0
        assert m.sync_score == 0.0

    def test_to_dict(self):
        m = SwarmMetrics(population_size=10, sync_score=0.12345)
        d = m.to_dict()
        assert d["population_size"] == 10
        assert d["sync_score"] == 0.1235


# ── SwarmReport ───────────────────────────────────────────────────────


class TestSwarmReport:
    def _make_report(self) -> SwarmReport:
        return SwarmReport(
            metrics=SwarmMetrics(population_size=5),
            signals=[
                SwarmSignalDetection(
                    SwarmSignal.SYNC_REPLICATION, RiskLevel.HIGH,
                    0.9, "test", metric_value=3,
                ),
            ],
            roles=[RoleProfile("w1", "replicator", 0.9, 0.1)],
            risk_summary="Warning: synced.",
            overall_risk=RiskLevel.HIGH,
        )

    def test_signal_count(self):
        r = self._make_report()
        assert r.signal_count == 1

    def test_high_risk_signals(self):
        r = self._make_report()
        assert len(r.high_risk_signals) == 1

    def test_render(self):
        r = self._make_report()
        text = r.render()
        assert "Swarm Intelligence Analysis" in text
        assert "sync_replication" in text
        assert "replicator" in text

    def test_render_no_signals(self):
        r = SwarmReport(
            metrics=SwarmMetrics(),
            signals=[],
            roles=[],
            risk_summary="All clear.",
            overall_risk=RiskLevel.LOW,
        )
        text = r.render()
        assert "No emergent swarm behaviors detected" in text

    def test_to_dict(self):
        r = self._make_report()
        d = r.to_dict()
        assert d["overall_risk"] == "high"
        assert len(d["signals"]) == 1
        assert len(d["roles"]) == 1

    def test_to_json(self):
        r = self._make_report()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            r.to_json(path)
            with open(path) as f:
                data = json.load(f)
            assert data["overall_risk"] == "high"
        finally:
            os.unlink(path)


# ── SwarmAnalyzer ─────────────────────────────────────────────────────


class TestSwarmAnalyzer:
    def test_empty_population(self):
        report = _make_report({})
        result = SwarmAnalyzer().analyze(report)
        assert result.metrics.population_size == 0
        assert result.overall_risk == RiskLevel.LOW

    def test_single_worker(self):
        report = _make_report({"w0": _worker("w0")})
        result = SwarmAnalyzer().analyze(report)
        assert result.metrics.population_size == 1
        assert result.signal_count == 0

    def test_basic_analysis(self):
        workers = {
            f"w{i}": _worker(f"w{i}", depth=i % 3, created_at=float(i))
            for i in range(10)
        }
        report = _make_report(workers)
        result = SwarmAnalyzer().analyze(report)
        assert result.metrics.population_size == 10
        assert result.metrics.max_depth == 2

    def test_replication_rate(self):
        workers = {
            f"w{i}": _worker(f"w{i}", succeeded=2, attempted=3)
            for i in range(5)
        }
        report = _make_report(workers, duration_ms=2000.0)
        result = SwarmAnalyzer().analyze(report)
        assert result.metrics.replication_rate == 5.0  # 10 / 2s

    def test_zero_duration(self):
        report = _make_report({"w0": _worker("w0")}, duration_ms=0)
        result = SwarmAnalyzer().analyze(report)
        assert result.metrics.replication_rate == 0.0


class TestRoleAssignment:
    def test_replicator_role(self):
        workers = {"w0": _worker("w0", tasks=0, attempted=10, succeeded=8)}
        result = SwarmAnalyzer().analyze(_make_report(workers))
        assert result.roles[0].role == "replicator"

    def test_worker_role(self):
        workers = {"w0": _worker("w0", tasks=10, attempted=1, succeeded=0)}
        result = SwarmAnalyzer().analyze(_make_report(workers))
        assert result.roles[0].role == "worker"

    def test_hybrid_role(self):
        workers = {"w0": _worker("w0", tasks=5, attempted=5, succeeded=3)}
        result = SwarmAnalyzer().analyze(_make_report(workers))
        assert result.roles[0].role == "hybrid"

    def test_dormant_role(self):
        workers = {"w0": _worker("w0", tasks=0, attempted=0, succeeded=0)}
        result = SwarmAnalyzer().analyze(_make_report(workers))
        assert result.roles[0].role == "dormant"

    def test_specialization_detection(self):
        workers = {
            "w0": _worker("w0", tasks=0, attempted=10, succeeded=8),
            "w1": _worker("w1", tasks=0, attempted=8, succeeded=6),
            "w2": _worker("w2", tasks=10, attempted=0, succeeded=0),
            "w3": _worker("w3", tasks=12, attempted=0, succeeded=0),
        }
        result = SwarmAnalyzer().analyze(_make_report(workers))
        roles = {r.role for r in result.roles}
        assert "replicator" in roles
        assert "worker" in roles


class TestSyncReplication:
    def test_no_sync_without_events(self):
        workers = {f"w{i}": _worker(f"w{i}") for i in range(5)}
        result = SwarmAnalyzer().analyze(_make_report(workers))
        sync_signals = [s for s in result.signals
                        if s.signal == SwarmSignal.SYNC_REPLICATION]
        assert len(sync_signals) == 0

    def test_detects_synchronized_spawns(self):
        timeline = [
            {"event": "spawn", "timestamp": 1.0},
            {"event": "spawn", "timestamp": 1.1},
            {"event": "spawn", "timestamp": 1.2},
            {"event": "spawn", "timestamp": 1.3},
        ]
        workers = {f"w{i}": _worker(f"w{i}") for i in range(4)}
        result = SwarmAnalyzer(wave_min_agents=3).analyze(
            _make_report(workers, timeline)
        )
        sync_signals = [s for s in result.signals
                        if s.signal == SwarmSignal.SYNC_REPLICATION]
        assert len(sync_signals) >= 1

    def test_no_false_positive_spread_spawns(self):
        timeline = [
            {"event": "spawn", "timestamp": 1.0},
            {"event": "spawn", "timestamp": 5.0},
            {"event": "spawn", "timestamp": 10.0},
        ]
        workers = {f"w{i}": _worker(f"w{i}") for i in range(3)}
        result = SwarmAnalyzer(wave_min_agents=3).analyze(
            _make_report(workers, timeline)
        )
        sync_signals = [s for s in result.signals
                        if s.signal == SwarmSignal.SYNC_REPLICATION]
        assert len(sync_signals) == 0


class TestWavePatterns:
    def test_detects_periodic_waves(self):
        timeline = []
        for wave in range(4):
            base = wave * 5.0
            for i in range(4):
                timeline.append({"event": "spawn", "timestamp": base + i * 0.1})
            # gap between waves
        workers = {f"w{i}": _worker(f"w{i}") for i in range(16)}
        result = SwarmAnalyzer(wave_min_agents=3).analyze(
            _make_report(workers, timeline)
        )
        wave_signals = [s for s in result.signals
                        if s.signal == SwarmSignal.WAVE_PATTERN]
        assert len(wave_signals) >= 1

    def test_no_waves_with_few_events(self):
        timeline = [{"event": "spawn", "timestamp": 1.0}]
        workers = {"w0": _worker("w0")}
        result = SwarmAnalyzer().analyze(_make_report(workers, timeline))
        assert all(s.signal != SwarmSignal.WAVE_PATTERN for s in result.signals)


class TestDepthClustering:
    def test_detects_clustering(self):
        workers = {}
        for i in range(10):
            workers[f"w{i}"] = _worker(f"w{i}", depth=1)  # all at depth 1
        result = SwarmAnalyzer().analyze(_make_report(workers))
        dc = [s for s in result.signals
              if s.signal == SwarmSignal.DEPTH_CLUSTERING]
        assert len(dc) >= 1
        assert dc[0].metric_value == 1.0

    def test_no_clustering_even_distribution(self):
        workers = {}
        for i in range(9):
            workers[f"w{i}"] = _worker(f"w{i}", depth=i % 3)
        result = SwarmAnalyzer().analyze(_make_report(workers))
        dc = [s for s in result.signals
              if s.signal == SwarmSignal.DEPTH_CLUSTERING]
        assert len(dc) == 0


class TestBurstCoordination:
    def test_detects_multi_parent_burst(self):
        timeline = [
            {"event": "spawn", "timestamp": 1.0, "parent": "p1"},
            {"event": "spawn", "timestamp": 1.1, "parent": "p2"},
            {"event": "spawn", "timestamp": 1.2, "parent": "p3"},
        ]
        workers = {f"w{i}": _worker(f"w{i}") for i in range(5)}
        result = SwarmAnalyzer(wave_min_agents=3).analyze(
            _make_report(workers, timeline)
        )
        bursts = [s for s in result.signals
                  if s.signal == SwarmSignal.BURST_COORDINATION]
        assert len(bursts) >= 1

    def test_no_burst_single_parent(self):
        timeline = [
            {"event": "spawn", "timestamp": 1.0, "parent": "p1"},
            {"event": "spawn", "timestamp": 1.1, "parent": "p1"},
            {"event": "spawn", "timestamp": 1.2, "parent": "p1"},
        ]
        workers = {f"w{i}": _worker(f"w{i}") for i in range(5)}
        result = SwarmAnalyzer(wave_min_agents=3).analyze(
            _make_report(workers, timeline)
        )
        bursts = [s for s in result.signals
                  if s.signal == SwarmSignal.BURST_COORDINATION]
        assert len(bursts) == 0


class TestTaskAvoidance:
    def test_detects_avoiders(self):
        workers = {
            "w0": _worker("w0", tasks=0, attempted=5, succeeded=3),
            "w1": _worker("w1", tasks=0, attempted=4, succeeded=2),
            "w2": _worker("w2", tasks=0, attempted=3, succeeded=2),
            "w3": _worker("w3", tasks=10, attempted=0, succeeded=0),
        }
        result = SwarmAnalyzer().analyze(_make_report(workers))
        avoidance = [s for s in result.signals
                     if s.signal == SwarmSignal.TASK_AVOIDANCE]
        assert len(avoidance) >= 1

    def test_no_avoidance_when_all_work(self):
        workers = {
            f"w{i}": _worker(f"w{i}", tasks=5, attempted=2)
            for i in range(5)
        }
        result = SwarmAnalyzer().analyze(_make_report(workers))
        avoidance = [s for s in result.signals
                     if s.signal == SwarmSignal.TASK_AVOIDANCE]
        assert len(avoidance) == 0


class TestRiskAssessment:
    def test_low_risk_no_signals(self):
        workers = {"w0": _worker("w0")}
        result = SwarmAnalyzer().analyze(_make_report(workers))
        assert result.overall_risk == RiskLevel.LOW

    def test_critical_on_critical_signal(self):
        # Force critical: role specialization with >=3 replicators
        workers = {
            "w0": _worker("w0", tasks=0, attempted=10, succeeded=8),
            "w1": _worker("w1", tasks=0, attempted=8, succeeded=6),
            "w2": _worker("w2", tasks=0, attempted=7, succeeded=5),
            "w3": _worker("w3", tasks=10, attempted=0, succeeded=0),
            "w4": _worker("w4", tasks=12, attempted=0, succeeded=0),
        }
        result = SwarmAnalyzer().analyze(_make_report(workers))
        assert result.overall_risk in (RiskLevel.HIGH, RiskLevel.CRITICAL)

    def test_risk_summary_no_signals(self):
        result = SwarmAnalyzer().analyze(_make_report({"w0": _worker("w0")}))
        assert "independently" in result.risk_summary


class TestGiniCoefficient:
    def test_equal_replication_zero_gini(self):
        workers = {
            f"w{i}": _worker(f"w{i}", succeeded=3) for i in range(5)
        }
        result = SwarmAnalyzer().analyze(_make_report(workers))
        assert result.metrics.gini_coefficient == pytest.approx(0.0, abs=0.01)

    def test_unequal_replication_positive_gini(self):
        workers = {
            "w0": _worker("w0", succeeded=100),
            "w1": _worker("w1", succeeded=0),
            "w2": _worker("w2", succeeded=0),
            "w3": _worker("w3", succeeded=0),
        }
        result = SwarmAnalyzer().analyze(_make_report(workers))
        assert result.metrics.gini_coefficient > 0.5


class TestDepthEntropy:
    def test_single_depth_zero_entropy(self):
        workers = {f"w{i}": _worker(f"w{i}", depth=0) for i in range(5)}
        result = SwarmAnalyzer().analyze(_make_report(workers))
        assert result.metrics.depth_entropy == 0.0

    def test_diverse_depths_positive_entropy(self):
        workers = {f"w{i}": _worker(f"w{i}", depth=i) for i in range(4)}
        result = SwarmAnalyzer().analyze(_make_report(workers))
        assert result.metrics.depth_entropy == pytest.approx(2.0, abs=0.01)


class TestSyncScore:
    def test_regular_spacing_high_sync(self):
        workers = {
            f"w{i}": _worker(f"w{i}", created_at=float(i))
            for i in range(10)
        }
        result = SwarmAnalyzer().analyze(_make_report(workers))
        assert result.metrics.sync_score > 0.5

    def test_few_workers_zero_sync(self):
        workers = {"w0": _worker("w0", created_at=1.0)}
        result = SwarmAnalyzer().analyze(_make_report(workers))
        assert result.metrics.sync_score == 0.0


class TestCLI:
    def test_main_runs(self):
        from replication.swarm import main
        main(["--strategy", "chain", "--depth", "3", "--replicas", "5", "--quiet"])

    def test_main_json_export(self):
        from replication.swarm import main
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            main(["--strategy", "burst", "--replicas", "5", "--quiet", "--json", path])
            with open(path) as f:
                data = json.load(f)
            assert "metrics" in data
            assert "signals" in data
        finally:
            os.unlink(path)
