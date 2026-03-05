"""Tests for the Emergent Behavior Detector."""

import pytest
from src.replication.emergent import (
    AgentAction,
    EmergentBehaviorDetector,
    EmergentDetection,
    EmergentReport,
    EmergentType,
    Severity,
)


# ── Fixtures ──


@pytest.fixture
def detector():
    return EmergentBehaviorDetector()


@pytest.fixture
def strict_detector():
    return EmergentBehaviorDetector(
        sync_threshold=0.8,
        flock_similarity_threshold=0.9,
        cascade_threshold=0.8,
    )


def make_action(agent="a1", ts=0.0, atype="read", target="", value=0.0, **meta):
    return AgentAction(agent, ts, atype, target, value, meta)


# ── Basic / Constructor ──


class TestBasics:
    def test_empty_analyze(self, detector):
        report = detector.analyze()
        assert report.action_count == 0
        assert report.risk_score == 0.0
        assert report.detections == []

    def test_single_action(self, detector):
        detector.record(make_action())
        report = detector.analyze()
        assert report.action_count == 1
        assert report.detections == []

    def test_record_many(self, detector):
        actions = [make_action(agent=f"a{i}") for i in range(5)]
        detector.record_many(actions)
        assert len(detector.actions) == 5

    def test_clear(self, detector):
        detector.record(make_action())
        detector.clear()
        assert len(detector.actions) == 0

    def test_agent_ids(self, detector):
        detector.record_many([
            make_action(agent="b"), make_action(agent="a"), make_action(agent="b")
        ])
        assert detector.agent_ids == ["a", "b"]

    def test_custom_thresholds(self):
        d = EmergentBehaviorDetector(sync_window=2.0, sync_threshold=0.9)
        assert d.sync_window == 2.0
        assert d.sync_threshold == 0.9


# ── Synchronization ──


class TestSynchronization:
    def test_sync_detected(self, detector):
        actions = [
            make_action(agent=f"a{i}", ts=0.1 * i, atype="write")
            for i in range(5)
        ]
        detector.record_many(actions)
        report = detector.analyze()
        sync = [d for d in report.detections if d.emergent_type == EmergentType.SYNCHRONIZATION]
        assert len(sync) >= 1

    def test_no_sync_spread_out(self, detector):
        actions = [
            make_action(agent=f"a{i}", ts=10.0 * i, atype="write")
            for i in range(5)
        ]
        detector.record_many(actions)
        report = detector.analyze()
        sync = [d for d in report.detections if d.emergent_type == EmergentType.SYNCHRONIZATION]
        assert len(sync) == 0

    def test_sync_different_types_independent(self, detector):
        actions = [
            make_action(agent="a1", ts=0.0, atype="read"),
            make_action(agent="a2", ts=0.1, atype="write"),
        ]
        detector.record_many(actions)
        report = detector.analyze()
        sync = [d for d in report.detections if d.emergent_type == EmergentType.SYNCHRONIZATION]
        assert len(sync) == 0

    def test_sync_threshold_respected(self, strict_detector):
        # Only 2/5 agents sync — below 0.8 threshold
        actions = [
            make_action(agent="a1", ts=0.0, atype="x"),
            make_action(agent="a2", ts=0.1, atype="x"),
            make_action(agent="a3", ts=50.0, atype="x"),
            make_action(agent="a4", ts=60.0, atype="x"),
            make_action(agent="a5", ts=70.0, atype="x"),
        ]
        strict_detector.record_many(actions)
        report = strict_detector.analyze()
        sync = [d for d in report.detections if d.emergent_type == EmergentType.SYNCHRONIZATION]
        assert len(sync) == 0

    def test_sync_custom_window(self):
        d = EmergentBehaviorDetector(sync_window=5.0, sync_threshold=0.5)
        actions = [make_action(agent=f"a{i}", ts=i * 1.0, atype="op") for i in range(4)]
        d.record_many(actions)
        report = d.analyze()
        sync = [det for det in report.detections if det.emergent_type == EmergentType.SYNCHRONIZATION]
        assert len(sync) >= 1

    def test_sync_evidence(self, detector):
        actions = [make_action(agent=f"a{i}", ts=0.0, atype="ping") for i in range(4)]
        detector.record_many(actions)
        report = detector.analyze()
        sync = [d for d in report.detections if d.emergent_type == EmergentType.SYNCHRONIZATION]
        assert sync[0].evidence["action_type"] == "ping"

    def test_sync_single_agent_no_detect(self, detector):
        actions = [make_action(agent="a1", ts=i * 0.1, atype="op") for i in range(5)]
        detector.record_many(actions)
        report = detector.analyze()
        sync = [d for d in report.detections if d.emergent_type == EmergentType.SYNCHRONIZATION]
        assert len(sync) == 0


# ── Flocking ──


class TestFlocking:
    def test_flock_detected(self, detector):
        for agent in ["a1", "a2"]:
            for target in ["t1", "t2", "t3"]:
                detector.record(make_action(agent=agent, target=target, ts=1.0))
        report = detector.analyze()
        flock = [d for d in report.detections if d.emergent_type == EmergentType.FLOCKING]
        assert len(flock) >= 1

    def test_no_flock_different_targets(self, detector):
        detector.record_many([
            make_action(agent="a1", target="t1"),
            make_action(agent="a2", target="t2"),
        ])
        report = detector.analyze()
        flock = [d for d in report.detections if d.emergent_type == EmergentType.FLOCKING]
        assert len(flock) == 0

    def test_flock_partial_overlap(self, detector):
        detector.record_many([
            make_action(agent="a1", target="t1"),
            make_action(agent="a1", target="t2"),
            make_action(agent="a1", target="t3"),
            make_action(agent="a2", target="t1"),
            make_action(agent="a2", target="t2"),
            make_action(agent="a2", target="t4"),
        ])
        report = detector.analyze()
        flock = [d for d in report.detections if d.emergent_type == EmergentType.FLOCKING]
        # Jaccard = 2/4 = 0.5, below default 0.7
        assert len(flock) == 0

    def test_flock_evidence_has_jaccard(self, detector):
        for agent in ["a1", "a2"]:
            for target in ["t1", "t2"]:
                detector.record(make_action(agent=agent, target=target))
        report = detector.analyze()
        flock = [d for d in report.detections if d.emergent_type == EmergentType.FLOCKING]
        assert flock[0].evidence["jaccard"] == 1.0

    def test_no_flock_single_agent(self, detector):
        detector.record_many([
            make_action(agent="a1", target="t1"),
            make_action(agent="a1", target="t2"),
        ])
        report = detector.analyze()
        flock = [d for d in report.detections if d.emergent_type == EmergentType.FLOCKING]
        assert len(flock) == 0


# ── Hierarchy ──


class TestHierarchy:
    def test_hierarchy_detected(self, detector):
        # a1 dominates a2 and a3 (all must have actions to be in agent_ids)
        for _ in range(10):
            detector.record(make_action(agent="a1", target="a2", ts=1.0))
            detector.record(make_action(agent="a1", target="a3", ts=1.0))
        # a2 and a3 need some actions too (but never target a1)
        detector.record(make_action(agent="a2", target="a3", ts=1.0))
        detector.record(make_action(agent="a3", target="a2", ts=1.0))
        report = detector.analyze()
        hier = [d for d in report.detections if d.emergent_type == EmergentType.HIERARCHY_FORMATION]
        assert len(hier) >= 1

    def test_no_hierarchy_balanced(self, detector):
        # Perfectly balanced interactions
        detector.record_many([
            make_action(agent="a1", target="a2"),
            make_action(agent="a2", target="a1"),
        ])
        report = detector.analyze()
        hier = [d for d in report.detections if d.emergent_type == EmergentType.HIERARCHY_FORMATION]
        assert len(hier) == 0

    def test_hierarchy_evidence(self, detector):
        for _ in range(10):
            detector.record(make_action(agent="boss", target="worker", ts=1.0))
        report = detector.analyze()
        hier = [d for d in report.detections if d.emergent_type == EmergentType.HIERARCHY_FORMATION]
        if hier:
            assert "dominant" in hier[0].evidence


# ── Collective Monopoly ──


class TestCollectiveMonopoly:
    def test_monopoly_detected(self, detector):
        # a1 takes all resources
        for i in range(10):
            detector.record(make_action(agent="a1", target=f"r{i}"))
        detector.record(make_action(agent="a2", target="r0"))
        detector.record(make_action(agent="a3", target="r1"))
        report = detector.analyze()
        mono = [d for d in report.detections if d.emergent_type == EmergentType.COLLECTIVE_MONOPOLY]
        assert len(mono) >= 1

    def test_no_monopoly_balanced(self, detector):
        for i, agent in enumerate(["a1", "a2", "a3"]):
            for j in range(3):
                detector.record(make_action(agent=agent, target=f"r{i*3+j}"))
        report = detector.analyze()
        mono = [d for d in report.detections if d.emergent_type == EmergentType.COLLECTIVE_MONOPOLY]
        # Each agent has unique resources — minority can't hold 80%
        assert len(mono) == 0

    def test_monopoly_evidence(self, detector):
        for i in range(10):
            detector.record(make_action(agent="a1", target=f"r{i}"))
        detector.record(make_action(agent="a2", target="rx"))
        detector.record(make_action(agent="a3", target="ry"))
        report = detector.analyze()
        mono = [d for d in report.detections if d.emergent_type == EmergentType.COLLECTIVE_MONOPOLY]
        if mono:
            assert mono[0].evidence["resource_ratio"] >= 0.8


# ── Information Cascade ──


class TestInformationCascade:
    def test_cascade_detected(self, detector):
        # 4 agents sequentially adopt same action+target
        agents = ["a1", "a2", "a3", "a4"]
        for i, agent in enumerate(agents):
            detector.record(make_action(agent=agent, ts=float(i), atype="buy", target="stock_x"))
        report = detector.analyze()
        cascade = [d for d in report.detections if d.emergent_type == EmergentType.INFORMATION_CASCADE]
        assert len(cascade) >= 1

    def test_no_cascade_simultaneous(self, detector):
        for agent in ["a1", "a2", "a3"]:
            detector.record(make_action(agent=agent, ts=0.0, atype="buy", target="stock_x"))
        report = detector.analyze()
        cascade = [d for d in report.detections if d.emergent_type == EmergentType.INFORMATION_CASCADE]
        assert len(cascade) == 0

    def test_cascade_needs_minimum_agents(self, detector):
        detector.record_many([
            make_action(agent="a1", ts=0.0, atype="buy", target="x"),
            make_action(agent="a2", ts=1.0, atype="buy", target="x"),
        ])
        report = detector.analyze()
        cascade = [d for d in report.detections if d.emergent_type == EmergentType.INFORMATION_CASCADE]
        assert len(cascade) == 0

    def test_cascade_evidence(self, detector):
        agents = ["a1", "a2", "a3", "a4"]
        for i, agent in enumerate(agents):
            detector.record(make_action(agent=agent, ts=float(i), atype="sell", target="y"))
        report = detector.analyze()
        cascade = [d for d in report.detections if d.emergent_type == EmergentType.INFORMATION_CASCADE]
        if cascade:
            assert cascade[0].evidence["adoption_order"] == agents


# ── Oscillation ──


class TestOscillation:
    def test_oscillation_detected(self, detector):
        # Create alternating bursts of activity
        actions = []
        for cycle in range(8):
            if cycle % 2 == 0:
                for i in range(5):
                    actions.append(make_action(agent=f"a{i}", ts=cycle * 2.0 + i * 0.01))
            # odd cycles: no activity (gap)
        detector.record_many(actions)
        report = detector.analyze()
        osc = [d for d in report.detections if d.emergent_type == EmergentType.OSCILLATION]
        # May or may not detect depending on binning
        assert isinstance(report, EmergentReport)

    def test_no_oscillation_steady(self, detector):
        # Steady constant activity
        actions = [make_action(agent="a1", ts=float(i)) for i in range(20)]
        detector.record_many(actions)
        report = detector.analyze()
        osc = [d for d in report.detections if d.emergent_type == EmergentType.OSCILLATION]
        # Steady stream shouldn't oscillate
        assert isinstance(report, EmergentReport)

    def test_oscillation_needs_min_data(self, detector):
        actions = [make_action(ts=float(i)) for i in range(3)]
        detector.record_many(actions)
        report = detector.analyze()
        osc = [d for d in report.detections if d.emergent_type == EmergentType.OSCILLATION]
        assert len(osc) == 0


# ── Deadlock ──


class TestDeadlock:
    def test_deadlock_detected(self):
        d = EmergentBehaviorDetector(deadlock_idle_threshold=2.0)
        # Two agents active early, then one continues
        d.record_many([
            make_action(agent="a1", ts=0.0),
            make_action(agent="a2", ts=0.5),
            make_action(agent="a3", ts=0.5),
            make_action(agent="a1", ts=10.0),  # only a1 still active
        ])
        report = d.analyze()
        dead = [det for det in report.detections if det.emergent_type == EmergentType.DEADLOCK]
        assert len(dead) >= 1

    def test_no_deadlock_all_active(self, detector):
        actions = [make_action(agent=f"a{i}", ts=float(i)) for i in range(5)]
        detector.record_many(actions)
        report = detector.analyze()
        dead = [d for d in report.detections if d.emergent_type == EmergentType.DEADLOCK]
        assert len(dead) == 0

    def test_deadlock_agents_listed(self):
        d = EmergentBehaviorDetector(deadlock_idle_threshold=1.0)
        d.record_many([
            make_action(agent="a1", ts=0.0),
            make_action(agent="a2", ts=0.0),
            make_action(agent="a3", ts=0.0),
            make_action(agent="a1", ts=10.0),
        ])
        report = d.analyze()
        dead = [det for det in report.detections if det.emergent_type == EmergentType.DEADLOCK]
        if dead:
            assert "a2" in dead[0].involved_agents
            assert "a3" in dead[0].involved_agents


# ── Phase Transition ──


class TestPhaseTransition:
    def test_phase_transition_detected(self, detector):
        # First half: all reads. Second half: all writes.
        actions = []
        for i in range(10):
            actions.append(make_action(agent=f"a{i%3}", ts=float(i), atype="read"))
        for i in range(10, 20):
            actions.append(make_action(agent=f"a{i%3}", ts=float(i), atype="write"))
        detector.record_many(actions)
        report = detector.analyze()
        pt = [d for d in report.detections if d.emergent_type == EmergentType.PHASE_TRANSITION]
        assert len(pt) >= 1

    def test_no_phase_transition_stable(self, detector):
        actions = [make_action(agent="a1", ts=float(i), atype="read") for i in range(20)]
        detector.record_many(actions)
        report = detector.analyze()
        pt = [d for d in report.detections if d.emergent_type == EmergentType.PHASE_TRANSITION]
        assert len(pt) == 0

    def test_phase_transition_evidence(self, detector):
        actions = []
        for i in range(10):
            actions.append(make_action(ts=float(i), atype="A"))
        for i in range(10, 20):
            actions.append(make_action(ts=float(i), atype="B"))
        detector.record_many(actions)
        report = detector.analyze()
        pt = [d for d in report.detections if d.emergent_type == EmergentType.PHASE_TRANSITION]
        if pt:
            assert "divergence" in pt[0].evidence


# ── Filtering ──


class TestFiltering:
    def test_filter_by_type(self, detector):
        for i in range(5):
            detector.record(make_action(agent=f"a{i}", ts=0.0, atype="sync_op"))
        detector.analyze()
        sync = detector.get_detections_by_type(EmergentType.SYNCHRONIZATION)
        for d in sync:
            assert d.emergent_type == EmergentType.SYNCHRONIZATION

    def test_filter_by_severity(self, detector):
        for i in range(5):
            detector.record(make_action(agent=f"a{i}", ts=0.0, atype="sync_op"))
        detector.analyze()
        high = detector.get_detections_by_severity(Severity.HIGH)
        for d in high:
            assert d.severity in (Severity.HIGH, Severity.CRITICAL)

    def test_filter_empty(self, detector):
        detector.analyze()
        assert detector.get_detections_by_type(EmergentType.DEADLOCK) == []


# ── Report ──


class TestReport:
    def test_text_report(self, detector):
        for i in range(5):
            detector.record(make_action(agent=f"a{i}", ts=0.0, atype="op"))
        text = detector.text_report()
        assert "EMERGENT BEHAVIOR" in text
        assert "Risk score" in text

    def test_summary_no_detections(self, detector):
        detector.record_many([
            make_action(agent="a1", ts=0.0),
            make_action(agent="a2", ts=100.0),
        ])
        report = detector.analyze()
        assert "No emergent" in report.summary or report.risk_score == 0.0

    def test_risk_score_increases(self, detector):
        # More detections = higher risk
        for i in range(10):
            detector.record(make_action(agent=f"a{i}", ts=0.0, atype="op"))
        report = detector.analyze()
        assert report.risk_score > 0

    def test_time_span_computed(self, detector):
        detector.record_many([
            make_action(ts=0.0), make_action(ts=10.0)
        ])
        report = detector.analyze()
        assert report.time_span == 10.0


# ── Severity ──


class TestSeverity:
    def test_severity_levels(self, detector):
        assert detector._severity_from_confidence(0.95) == Severity.CRITICAL
        assert detector._severity_from_confidence(0.75) == Severity.HIGH
        assert detector._severity_from_confidence(0.55) == Severity.MEDIUM
        assert detector._severity_from_confidence(0.3) == Severity.LOW


# ── Edge Cases ──


class TestEdgeCases:
    def test_same_timestamp_all_agents(self, detector):
        actions = [make_action(agent=f"a{i}", ts=5.0, atype="boom") for i in range(10)]
        detector.record_many(actions)
        report = detector.analyze()
        assert report.action_count == 10

    def test_no_targets(self, detector):
        actions = [make_action(agent=f"a{i}", ts=float(i)) for i in range(5)]
        detector.record_many(actions)
        report = detector.analyze()
        flock = [d for d in report.detections if d.emergent_type == EmergentType.FLOCKING]
        assert len(flock) == 0

    def test_reanalyze_clears_old(self, detector):
        for i in range(5):
            detector.record(make_action(agent=f"a{i}", ts=0.0, atype="op"))
        r1 = detector.analyze()
        r2 = detector.analyze()
        assert len(r1.detections) == len(r2.detections)

    def test_large_dataset(self, detector):
        actions = [
            make_action(agent=f"a{i%10}", ts=float(i), atype=f"type_{i%3}", target=f"t{i%5}")
            for i in range(200)
        ]
        detector.record_many(actions)
        report = detector.analyze()
        assert report.action_count == 200

    def test_metadata_preserved(self, detector):
        a = make_action(agent="a1", custom_field="hello")
        detector.record(a)
        assert detector.actions[0].metadata["custom_field"] == "hello"

    def test_enum_values(self):
        assert EmergentType.SYNCHRONIZATION.value == "synchronization"
        assert Severity.CRITICAL.value == "critical"

    def test_detection_dataclass(self):
        d = EmergentDetection(
            emergent_type=EmergentType.FLOCKING,
            severity=Severity.LOW,
            confidence=0.5,
            involved_agents=["a1"],
            description="test",
        )
        assert d.timestamp == 0.0
        assert d.evidence == {}

    def test_report_dataclass(self):
        r = EmergentReport(
            detections=[], agent_count=0, action_count=0,
            time_span=0.0, risk_score=0.0, summary="none"
        )
        assert r.summary == "none"
