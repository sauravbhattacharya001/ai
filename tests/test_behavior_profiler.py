"""Tests for the behavior profiler module."""

from __future__ import annotations

import pytest

from replication.behavior_profiler import (
    Action,
    ActionCategory,
    AgentBaseline,
    Anomaly,
    AnomalySeverity,
    AnomalyType,
    BehaviorProfiler,
    FleetReport,
    ProfileReport,
    ProfilerConfig,
    SEVERITY_WEIGHT,
    _jensen_shannon_divergence,
    _risk_level,
    _shannon_entropy,
    demo,
)


# ── Fixtures ─────────────────────────────────────────────────────


def _make_actions(
    agent_id: str = "agent-0",
    categories: list[ActionCategory] | None = None,
    resources: list[str] | None = None,
    count: int = 20,
    start_ts: float = 1000.0,
    interval: float = 10.0,
) -> list[Action]:
    """Create a list of synthetic actions."""
    if categories is None:
        categories = [ActionCategory.COMPUTE, ActionCategory.STORAGE, ActionCategory.NETWORK]
    if resources is None:
        resources = ["res-a", "res-b", "res-c"]
    actions = []
    for i in range(count):
        actions.append(
            Action(
                agent_id=agent_id,
                category=categories[i % len(categories)],
                resource=resources[i % len(resources)],
                timestamp=start_ts + i * interval,
            )
        )
    return actions


@pytest.fixture
def profiler() -> BehaviorProfiler:
    return BehaviorProfiler()


@pytest.fixture
def trained_profiler() -> BehaviorProfiler:
    p = BehaviorProfiler()
    training = _make_actions(count=50)
    p.train(training)
    return p


# ── Action Tests ─────────────────────────────────────────────────


class TestAction:
    def test_create_valid(self):
        a = Action(
            agent_id="a1",
            category=ActionCategory.COMPUTE,
            resource="cpu",
            timestamp=100.0,
        )
        assert a.agent_id == "a1"
        assert a.category == ActionCategory.COMPUTE
        assert a.timestamp == 100.0

    def test_empty_agent_id_raises(self):
        with pytest.raises(ValueError, match="agent_id"):
            Action(agent_id="", category=ActionCategory.COMPUTE, resource="x", timestamp=0)

    def test_negative_timestamp_raises(self):
        with pytest.raises(ValueError, match="timestamp"):
            Action(agent_id="a", category=ActionCategory.COMPUTE, resource="x", timestamp=-1)

    def test_frozen(self):
        a = Action(agent_id="a", category=ActionCategory.COMPUTE, resource="x", timestamp=0)
        with pytest.raises(AttributeError):
            a.agent_id = "b"  # type: ignore[misc]


# ── ProfilerConfig Tests ────────────────────────────────────────


class TestProfilerConfig:
    def test_defaults(self):
        c = ProfilerConfig()
        assert c.frequency_z_threshold == 2.5
        assert c.min_training_actions == 10
        assert c.burst_count_threshold == 10

    def test_invalid_z_threshold(self):
        with pytest.raises(ValueError):
            ProfilerConfig(frequency_z_threshold=0)

    def test_invalid_min_training(self):
        with pytest.raises(ValueError):
            ProfilerConfig(min_training_actions=0)

    def test_invalid_dormancy(self):
        with pytest.raises(ValueError):
            ProfilerConfig(dormancy_threshold=0)

    def test_invalid_burst(self):
        with pytest.raises(ValueError):
            ProfilerConfig(burst_count_threshold=1)


# ── Training Tests ───────────────────────────────────────────────


class TestTraining:
    def test_train_builds_baselines(self, profiler: BehaviorProfiler):
        actions = _make_actions(count=30)
        baselines = profiler.train(actions)
        assert "agent-0" in baselines
        assert baselines["agent-0"].action_count == 30

    def test_train_empty_raises(self, profiler: BehaviorProfiler):
        with pytest.raises(ValueError, match="empty"):
            profiler.train([])

    def test_train_skips_small(self, profiler: BehaviorProfiler):
        # Only 5 actions, below default min_training_actions=10
        actions = _make_actions(count=5)
        baselines = profiler.train(actions)
        assert "agent-0" not in baselines

    def test_train_multiple_agents(self, profiler: BehaviorProfiler):
        a1 = _make_actions(agent_id="a1", count=20)
        a2 = _make_actions(agent_id="a2", count=20, start_ts=2000.0)
        baselines = profiler.train(a1 + a2)
        assert "a1" in baselines
        assert "a2" in baselines

    def test_baseline_category_distribution(self, profiler: BehaviorProfiler):
        # All COMPUTE actions
        actions = _make_actions(
            categories=[ActionCategory.COMPUTE], count=20
        )
        baselines = profiler.train(actions)
        bl = baselines["agent-0"]
        assert bl.category_distribution[ActionCategory.COMPUTE] == 1.0

    def test_baseline_known_resources(self, profiler: BehaviorProfiler):
        actions = _make_actions(resources=["alpha", "beta"], count=20)
        baselines = profiler.train(actions)
        bl = baselines["agent-0"]
        assert bl.known_resources == frozenset(["alpha", "beta"])

    def test_baseline_interval_stats(self, profiler: BehaviorProfiler):
        actions = _make_actions(count=20, interval=10.0)
        baselines = profiler.train(actions)
        bl = baselines["agent-0"]
        assert bl.interval_mean == pytest.approx(10.0, abs=0.1)
        assert bl.interval_std == pytest.approx(0.0, abs=0.1)


# ── Analysis Tests ───────────────────────────────────────────────


class TestAnalysis:
    def test_analyze_empty_raises(self, trained_profiler: BehaviorProfiler):
        with pytest.raises(ValueError, match="empty"):
            trained_profiler.analyze([])

    def test_analyze_returns_fleet_report(self, trained_profiler: BehaviorProfiler):
        obs = _make_actions(count=20, start_ts=2000.0)
        report = trained_profiler.analyze(obs)
        assert isinstance(report, FleetReport)
        assert report.total_agents == 1

    def test_no_anomalies_for_normal(self, trained_profiler: BehaviorProfiler):
        # Same distribution as training → should have few/no anomalies
        obs = _make_actions(count=20, start_ts=2000.0)
        report = trained_profiler.analyze(obs)
        pr = report.agent_reports[0]
        # Risk should be low
        assert pr.risk_level in ("none", "low", "medium")

    def test_unknown_agent(self, trained_profiler: BehaviorProfiler):
        obs = _make_actions(agent_id="unknown", count=10, start_ts=2000.0)
        report = trained_profiler.analyze(obs)
        pr = report.agent_reports[0]
        assert pr.baseline_available is False
        assert pr.risk_level == "unknown"


# ── Anomaly Detection Tests ─────────────────────────────────────


class TestNewCategory:
    def test_detects_new_category(self, trained_profiler: BehaviorProfiler):
        # Training used COMPUTE, STORAGE, NETWORK — inject AUTH
        obs = [
            Action("agent-0", ActionCategory.AUTH, "creds", 2000.0 + i * 10)
            for i in range(15)
        ]
        report = trained_profiler.analyze(obs)
        pr = report.agent_reports[0]
        new_cat = [
            a for a in pr.anomalies if a.anomaly_type == AnomalyType.NEW_CATEGORY
        ]
        assert len(new_cat) >= 1
        assert new_cat[0].details["category"] == "auth"

    def test_auth_is_critical(self, trained_profiler: BehaviorProfiler):
        obs = [
            Action("agent-0", ActionCategory.AUTH, "creds", 2000.0 + i * 10)
            for i in range(15)
        ]
        report = trained_profiler.analyze(obs)
        pr = report.agent_reports[0]
        auth_anomalies = [
            a
            for a in pr.anomalies
            if a.anomaly_type == AnomalyType.NEW_CATEGORY
            and a.details.get("category") == "auth"
        ]
        assert auth_anomalies[0].severity == AnomalySeverity.CRITICAL


class TestNewResource:
    def test_detects_new_resources(self, trained_profiler: BehaviorProfiler):
        obs = [
            Action("agent-0", ActionCategory.COMPUTE, f"new-res-{i}", 2000.0 + i * 10)
            for i in range(15)
        ]
        report = trained_profiler.analyze(obs)
        pr = report.agent_reports[0]
        res_anomalies = [
            a
            for a in pr.anomalies
            if a.anomaly_type in (AnomalyType.NEW_RESOURCE, AnomalyType.RESOURCE_BREADTH)
        ]
        assert len(res_anomalies) >= 1


class TestBurstDetection:
    def test_detects_burst(self):
        config = ProfilerConfig(burst_count_threshold=5, burst_window_sec=2.0)
        profiler = BehaviorProfiler(config=config)
        training = _make_actions(count=50, interval=10.0)
        profiler.train(training)

        # Create burst: 10 actions in 1 second
        burst = [
            Action("agent-0", ActionCategory.COMPUTE, "res-a", 2000.0 + i * 0.1)
            for i in range(10)
        ]
        report = profiler.analyze(burst)
        pr = report.agent_reports[0]
        bursts = [
            a for a in pr.anomalies if a.anomaly_type == AnomalyType.BURST_DETECTED
        ]
        assert len(bursts) >= 1


class TestDormancy:
    def test_detects_dormancy_break(self):
        config = ProfilerConfig(dormancy_threshold=100.0)
        profiler = BehaviorProfiler(config=config)
        training = _make_actions(count=20, start_ts=1000.0, interval=10.0)
        profiler.train(training)

        # Resume after 500s gap (threshold is 100s)
        obs = _make_actions(count=10, start_ts=2000.0, interval=10.0)
        report = profiler.analyze(obs)
        pr = report.agent_reports[0]
        dormancy = [
            a for a in pr.anomalies if a.anomaly_type == AnomalyType.DORMANCY_BREAK
        ]
        assert len(dormancy) >= 1


class TestEntropy:
    def test_detects_entropy_shift(self):
        config = ProfilerConfig(entropy_shift_threshold=0.3)
        profiler = BehaviorProfiler(config=config)
        # Training: uniform across 4 categories (high entropy)
        training = _make_actions(
            categories=[
                ActionCategory.COMPUTE,
                ActionCategory.STORAGE,
                ActionCategory.NETWORK,
                ActionCategory.IPC,
            ],
            count=40,
        )
        profiler.train(training)

        # Analysis: all one category (zero entropy)
        obs = _make_actions(
            categories=[ActionCategory.COMPUTE],
            count=20,
            start_ts=2000.0,
        )
        report = profiler.analyze(obs)
        pr = report.agent_reports[0]
        entropy = [
            a for a in pr.anomalies if a.anomaly_type == AnomalyType.ENTROPY_SHIFT
        ]
        assert len(entropy) >= 1


# ── Risk Scoring ─────────────────────────────────────────────────


class TestRiskScoring:
    def test_zero_for_no_anomalies(self, trained_profiler: BehaviorProfiler):
        obs = _make_actions(count=15, start_ts=2000.0)
        report = trained_profiler.analyze(obs)
        pr = report.agent_reports[0]
        if not pr.anomalies:
            assert pr.risk_score == 0.0

    def test_high_for_critical_anomalies(self, trained_profiler: BehaviorProfiler):
        obs = [
            Action("agent-0", ActionCategory.AUTH, "vault", 2000.0 + i * 10)
            for i in range(20)
        ] + [
            Action("agent-0", ActionCategory.REPLICATION, "clone", 2200.0 + i * 10)
            for i in range(20)
        ]
        report = trained_profiler.analyze(obs)
        pr = report.agent_reports[0]
        assert pr.risk_score > 20

    def test_risk_level_mapping(self):
        assert _risk_level(0) == "none"
        assert _risk_level(10) == "low"
        assert _risk_level(30) == "medium"
        assert _risk_level(60) == "high"
        assert _risk_level(80) == "critical"


# ── Fleet Report ─────────────────────────────────────────────────


class TestFleetReport:
    def test_multi_agent_fleet(self):
        profiler = BehaviorProfiler()
        t1 = _make_actions(agent_id="a1", count=30)
        t2 = _make_actions(agent_id="a2", count=30, start_ts=2000.0)
        profiler.train(t1 + t2)

        obs = (
            _make_actions(agent_id="a1", count=10, start_ts=3000.0)
            + [
                Action("a2", ActionCategory.AUTH, "secret", 3000.0 + i * 10)
                for i in range(15)
            ]
        )
        report = profiler.analyze(obs)
        assert report.total_agents == 2
        assert report.total_anomalies >= 1

    def test_fleet_recommendations(self):
        profiler = BehaviorProfiler()
        training = _make_actions(count=30)
        profiler.train(training)
        obs = _make_actions(count=10, start_ts=2000.0)
        report = profiler.analyze(obs)
        assert len(report.fleet_recommendations) >= 1


# ── Utility Functions ────────────────────────────────────────────


class TestUtilities:
    def test_shannon_entropy_uniform(self):
        # Uniform over 4 categories → log2(4) = 2.0
        dist = {f"c{i}": 0.25 for i in range(4)}
        assert _shannon_entropy(dist) == pytest.approx(2.0, abs=0.01)

    def test_shannon_entropy_single(self):
        # All probability on one → entropy = 0
        assert _shannon_entropy({"a": 1.0}) == pytest.approx(0.0)

    def test_jsd_identical(self):
        d = {"a": 0.5, "b": 0.5}
        assert _jensen_shannon_divergence(d, d) == pytest.approx(0.0)

    def test_jsd_different(self):
        p = {"a": 1.0}
        q = {"b": 1.0}
        jsd = _jensen_shannon_divergence(p, q)
        assert jsd > 0

    def test_severity_weights(self):
        assert SEVERITY_WEIGHT[AnomalySeverity.CRITICAL] > SEVERITY_WEIGHT[AnomalySeverity.LOW]

    def test_anomaly_score_clamped(self):
        a = Anomaly(
            agent_id="x",
            anomaly_type=AnomalyType.BURST_DETECTED,
            severity=AnomalySeverity.LOW,
            score=5.0,  # should be clamped to 1.0
            description="test",
        )
        assert a.score == 1.0


# ── ProfileReport Properties ────────────────────────────────────


class TestProfileReport:
    def test_has_anomalies(self):
        pr = ProfileReport(
            agent_id="x",
            anomalies=[
                Anomaly("x", AnomalyType.BURST_DETECTED, AnomalySeverity.LOW, 0.5, "test")
            ],
            risk_score=10.0,
            risk_level="low",
            action_count=10,
            baseline_available=True,
            recommendations=[],
            summary={},
        )
        assert pr.has_anomalies is True

    def test_no_anomalies(self):
        pr = ProfileReport(
            agent_id="x",
            anomalies=[],
            risk_score=0.0,
            risk_level="none",
            action_count=10,
            baseline_available=True,
            recommendations=[],
            summary={},
        )
        assert pr.has_anomalies is False


# ── Recommendations ──────────────────────────────────────────────


class TestRecommendations:
    def test_auth_urgency(self, trained_profiler: BehaviorProfiler):
        obs = [
            Action("agent-0", ActionCategory.AUTH, "creds", 2000.0 + i * 10)
            for i in range(20)
        ]
        report = trained_profiler.analyze(obs)
        pr = report.agent_reports[0]
        assert any("URGENT" in r or "auth" in r.lower() for r in pr.recommendations)

    def test_burst_recommendation(self):
        config = ProfilerConfig(burst_count_threshold=5, burst_window_sec=2.0)
        profiler = BehaviorProfiler(config=config)
        profiler.train(_make_actions(count=50, interval=10.0))
        burst = [
            Action("agent-0", ActionCategory.COMPUTE, "res-a", 2000.0 + i * 0.1)
            for i in range(10)
        ]
        report = profiler.analyze(burst)
        pr = report.agent_reports[0]
        assert any("rate limit" in r.lower() for r in pr.recommendations)


# ── Demo ─────────────────────────────────────────────────────────


class TestDemo:
    def test_demo_runs(self):
        report = demo(num_agents=2, num_training=30, num_analysis=15)
        assert isinstance(report, FleetReport)
        assert report.total_agents >= 1

    def test_demo_detects_anomalies(self):
        report = demo(num_agents=3, num_training=100, num_analysis=30)
        assert report.total_anomalies >= 1

    def test_demo_reproducible(self):
        r1 = demo(seed=123)
        r2 = demo(seed=123)
        assert r1.total_anomalies == r2.total_anomalies
        assert r1.highest_risk_score == r2.highest_risk_score
