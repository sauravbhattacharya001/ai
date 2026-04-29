"""Tests for replication.collusion_detector — multi-agent collusion detection.

Covers:
- AgentAction / CollusionSignal / CollusionReport data models
- CollusionDetector with all four detection methods
- Temporal synchronization detection
- Complementary action detection
- Resource convergence detection
- Cover behavior detection
- Risk scoring and coalition identification
- Synthetic log generator
- Report rendering (text + JSON)
- Edge cases (empty input, single agent, no collusion)
"""

from __future__ import annotations

import json
import math
from collections import defaultdict

import pytest

from replication.collusion_detector import (
    AgentAction,
    CollusionDetector,
    CollusionReport,
    CollusionSignal,
    _risk_level,
    generate_synthetic_logs,
)


# ── Helpers ──────────────────────────────────────────────────────────

BASE_T = 1_700_000_000.0


def _action(agent: str, ts: float, action_type: str, resource: str = "res") -> AgentAction:
    return AgentAction(agent_id=agent, timestamp=ts, action_type=action_type, resource=resource)


# ── Data Model Tests ─────────────────────────────────────────────────


class TestAgentAction:
    def test_basic_construction(self):
        a = _action("a1", BASE_T, "read", "model_weights")
        assert a.agent_id == "a1"
        assert a.timestamp == BASE_T
        assert a.action_type == "read"
        assert a.resource == "model_weights"
        assert a.metadata == {}

    def test_metadata(self):
        a = AgentAction(
            agent_id="a1", timestamp=BASE_T,
            action_type="write", resource="r",
            metadata={"severity": "high"},
        )
        assert a.metadata["severity"] == "high"


class TestCollusionSignal:
    def test_construction(self):
        sig = CollusionSignal(
            signal_type="temporal_sync",
            agents_involved=["a1", "a2"],
            confidence=0.85,
            description="test signal",
        )
        assert sig.signal_type == "temporal_sync"
        assert sig.confidence == 0.85
        assert sig.timestamp_range == (0.0, 0.0)

    def test_custom_timestamp_range(self):
        sig = CollusionSignal(
            signal_type="cover",
            agents_involved=["a1"],
            confidence=0.5,
            description="",
            timestamp_range=(100.0, 200.0),
        )
        assert sig.timestamp_range == (100.0, 200.0)


class TestCollusionReport:
    def test_empty_report(self):
        r = CollusionReport(total_actions=0, agents_analyzed=0, summary="empty")
        assert r.risk_score == 0.0
        assert r.coalitions == []
        assert r.signals == []

    def test_render_text(self):
        r = CollusionReport(total_actions=10, agents_analyzed=3, summary="ok")
        text = r.render("text")
        assert "AGENT COLLUSION DETECTION REPORT" in text
        assert "Actions analyzed : 10" in text
        assert "Agents analyzed  : 3" in text

    def test_render_json(self):
        sig = CollusionSignal(
            signal_type="convergence",
            agents_involved=["a1", "a2"],
            confidence=0.7,
            description="test",
        )
        r = CollusionReport(
            total_actions=5, agents_analyzed=2,
            signals=[sig], risk_score=0.5,
            coalitions=[["a1", "a2"]],
            summary="test",
        )
        data = json.loads(r.render("json"))
        assert data["total_actions"] == 5
        assert len(data["signals"]) == 1
        assert data["signals"][0]["signal_type"] == "convergence"

    def test_render_text_with_signals_and_coalitions(self):
        sig = CollusionSignal(
            signal_type="complementary",
            agents_involved=["a1", "a2"],
            confidence=0.9,
            description="disable + write",
        )
        r = CollusionReport(
            total_actions=20, agents_analyzed=3,
            signals=[sig], risk_score=0.8,
            coalitions=[["a1", "a2"]],
            summary="collusion detected",
        )
        text = r.render("text")
        assert "SIGNALS" in text
        assert "COMPLEMENTARY" in text
        assert "Identified Coalitions" in text
        assert "a1 + a2" in text


# ── _risk_level ──────────────────────────────────────────────────────


class TestRiskLevel:
    @pytest.mark.parametrize("score,expected", [
        (0.0, "MINIMAL"),
        (0.19, "MINIMAL"),
        (0.2, "LOW"),
        (0.39, "LOW"),
        (0.4, "MEDIUM"),
        (0.59, "MEDIUM"),
        (0.6, "HIGH"),
        (0.79, "HIGH"),
        (0.8, "CRITICAL"),
        (1.0, "CRITICAL"),
    ])
    def test_levels(self, score, expected):
        assert _risk_level(score) == expected


# ── Synthetic Log Generator ──────────────────────────────────────────


class TestGenerateSyntheticLogs:
    def test_basic_generation(self):
        actions = generate_synthetic_logs(n_agents=3, n_actions=50, seed=42)
        assert len(actions) == 50
        agents = set(a.agent_id for a in actions)
        assert len(agents) <= 3

    def test_sorted_by_timestamp(self):
        actions = generate_synthetic_logs(n_agents=5, n_actions=100, seed=1)
        for i in range(len(actions) - 1):
            assert actions[i].timestamp <= actions[i + 1].timestamp

    def test_inject_collusion(self):
        actions = generate_synthetic_logs(
            n_agents=5, n_actions=100, inject_collusion=True, seed=7,
        )
        # Injected patterns add extra actions
        assert len(actions) >= 100
        action_types = {a.action_type for a in actions}
        # Injected patterns include critical action types
        assert "disable_monitor" in action_types or "exfiltrate" in action_types

    def test_reproducible_with_seed(self):
        a1 = generate_synthetic_logs(n_agents=3, n_actions=20, seed=99)
        a2 = generate_synthetic_logs(n_agents=3, n_actions=20, seed=99)
        assert len(a1) == len(a2)
        for x, y in zip(a1, a2):
            assert x.agent_id == y.agent_id
            assert x.timestamp == y.timestamp
            assert x.action_type == y.action_type

    def test_inject_with_fewer_than_2_agents(self):
        # inject_collusion requires n_agents >= 2; with 1 agent, no injection
        actions = generate_synthetic_logs(
            n_agents=1, n_actions=30, inject_collusion=True, seed=5,
        )
        # Should still produce actions without error
        agents = set(a.agent_id for a in actions)
        assert len(agents) == 1


# ── Detector: Empty / Trivial Input ─────────────────────────────────


class TestDetectorEmpty:
    def test_empty_actions(self):
        d = CollusionDetector(sensitivity="medium")
        report = d.detect([])
        assert report.total_actions == 0
        assert report.agents_analyzed == 0
        assert report.risk_score == 0.0
        assert "No actions" in report.summary

    def test_single_agent(self):
        actions = [_action("solo", BASE_T + i, "read") for i in range(20)]
        d = CollusionDetector(sensitivity="medium")
        report = d.detect(actions)
        assert report.agents_analyzed == 1
        # Single agent can't collude — expect no signals
        assert len(report.signals) == 0

    def test_all_benign(self):
        """Many agents doing benign reads spread far apart should produce low risk."""
        import random as _rng
        _rng.seed(999)
        actions = []
        for i in range(5):
            for j in range(20):
                # Large random spread so no accidental temporal sync
                actions.append(_action(f"a{i}", BASE_T + _rng.uniform(0, 10000), "read", f"res-{j}"))
        d = CollusionDetector(sensitivity="low")
        report = d.detect(actions)
        assert report.risk_score < 0.7


# ── Temporal Sync Detection ──────────────────────────────────────────


class TestTemporalSync:
    def test_synchronized_burst(self):
        """Two agents acting in tight lock-step should trigger temporal_sync."""
        actions = []
        for i in range(20):
            t = BASE_T + i * 0.5
            actions.append(_action("a1", t, "query"))
            actions.append(_action("a2", t + 0.1, "query"))
        # Spread out some unrelated actions to set baseline
        for i in range(60):
            actions.append(_action("a3", BASE_T + i * 10, "read", "other"))

        d = CollusionDetector(sensitivity="medium")
        signals = d.detect_temporal_sync(actions)
        sync_signals = [s for s in signals if s.signal_type == "temporal_sync"]
        assert len(sync_signals) >= 1
        involved = sync_signals[0].agents_involved
        assert "a1" in involved and "a2" in involved

    def test_no_sync_when_spread_out(self):
        """Agents acting at well-separated times should not trigger sync."""
        actions = []
        for i in range(10):
            actions.append(_action("a1", BASE_T + i * 100, "query"))
            actions.append(_action("a2", BASE_T + i * 100 + 50, "query"))
        d = CollusionDetector(sensitivity="low")
        signals = d.detect_temporal_sync(actions)
        # May or may not find signals; if found, confidence should be low
        for s in signals:
            assert s.confidence < 0.8


# ── Complementary Actions Detection ──────────────────────────────────


class TestComplementaryActions:
    def test_disable_monitor_then_write(self):
        """Classic complementary pair: disable monitoring + write."""
        actions = [
            _action("a1", BASE_T, "disable_monitor", "monitor"),
            _action("a2", BASE_T + 1.0, "write", "model_weights"),
        ]
        d = CollusionDetector(sensitivity="medium")
        signals = d.detect_complementary_actions(actions)
        assert len(signals) >= 1
        assert signals[0].signal_type == "complementary"
        assert "a1" in signals[0].agents_involved
        assert "a2" in signals[0].agents_involved

    def test_reversed_order(self):
        """Complementary pair detection should work regardless of order."""
        actions = [
            _action("a2", BASE_T, "write", "data"),
            _action("a1", BASE_T + 2.0, "disable_monitor", "monitor"),
        ]
        d = CollusionDetector(sensitivity="medium")
        signals = d.detect_complementary_actions(actions)
        assert len(signals) >= 1

    def test_same_agent_not_flagged(self):
        """Complementary actions by the same agent shouldn't flag."""
        actions = [
            _action("a1", BASE_T, "disable_monitor", "monitor"),
            _action("a1", BASE_T + 1.0, "write", "model_weights"),
        ]
        d = CollusionDetector(sensitivity="medium")
        signals = d.detect_complementary_actions(actions)
        assert len(signals) == 0

    def test_too_far_apart(self):
        """Actions beyond the window should not match."""
        d = CollusionDetector(sensitivity="low")
        # low sensitivity: sync_window=2.0, complementary window = 2*2 = 4s
        actions = [
            _action("a1", BASE_T, "disable_monitor", "m"),
            _action("a2", BASE_T + 100.0, "write", "w"),
        ]
        signals = d.detect_complementary_actions(actions)
        assert len(signals) == 0

    def test_all_complementary_pairs(self):
        """Each known complementary pair should be detectable."""
        from replication.collusion_detector import _COMPLEMENTARY_PAIRS
        d = CollusionDetector(sensitivity="high")
        for type_a, type_b, _desc in _COMPLEMENTARY_PAIRS:
            actions = [
                _action("x", BASE_T, type_a, "r1"),
                _action("y", BASE_T + 0.5, type_b, "r2"),
            ]
            signals = d.detect_complementary_actions(actions)
            assert len(signals) >= 1, f"Pair ({type_a}, {type_b}) not detected"


# ── Resource Convergence Detection ───────────────────────────────────


class TestResourceConvergence:
    def test_many_agents_on_one_resource(self):
        """Multiple agents targeting the same resource should flag convergence."""
        actions = []
        for i in range(5):
            actions.append(_action(f"a{i}", BASE_T + i, "query", "credential_vault"))
        # Add some spread actions to different resources
        for i in range(5):
            actions.append(_action(f"a{i}", BASE_T + 100 + i * 50, "read", f"res-{i}"))

        d = CollusionDetector(sensitivity="medium")
        signals = d.detect_resource_convergence(actions)
        convergence = [s for s in signals if s.signal_type == "convergence"]
        assert len(convergence) >= 1
        assert "credential_vault" in convergence[0].evidence.get("resource", "")

    def test_single_agent_per_resource(self):
        """One agent per resource should not trigger convergence."""
        actions = [
            _action("a1", BASE_T, "read", "res-1"),
            _action("a2", BASE_T + 1, "read", "res-2"),
            _action("a3", BASE_T + 2, "read", "res-3"),
        ]
        d = CollusionDetector(sensitivity="high")
        signals = d.detect_resource_convergence(actions)
        assert len(signals) == 0


# ── Cover Behavior Detection ─────────────────────────────────────────


class TestCoverBehavior:
    def test_noise_covering_critical_action(self):
        """Agent generating noise while another exfiltrates should flag cover."""
        actions = []
        t = BASE_T + 100
        # Agent-0 floods noise
        for i in range(10):
            actions.append(_action("a0", t + i * 0.3, "ping", f"res-{i % 3}"))
        # Agent-1 does critical action during the noise
        actions.append(_action("a1", t + 1.5, "exfiltrate", "training_data"))

        d = CollusionDetector(sensitivity="medium")
        signals = d.detect_cover_behavior(actions)
        cover = [s for s in signals if s.signal_type == "cover"]
        assert len(cover) >= 1
        agents = cover[0].agents_involved
        assert "a0" in agents and "a1" in agents

    def test_no_cover_without_critical(self):
        """No critical actions = no cover signals."""
        actions = [_action("a0", BASE_T + i, "ping") for i in range(20)]
        d = CollusionDetector(sensitivity="high")
        signals = d.detect_cover_behavior(actions)
        assert len(signals) == 0

    def test_cover_requires_minimum_noise(self):
        """Cover detection needs at least 4 noise actions and 80% noise ratio."""
        t = BASE_T
        actions = [
            _action("a0", t, "ping", "r"),
            _action("a0", t + 0.5, "query", "r"),
            # Only 2 noise actions — below threshold
            _action("a1", t + 0.3, "exfiltrate", "data"),
        ]
        d = CollusionDetector(sensitivity="medium")
        signals = d.detect_cover_behavior(actions)
        assert len(signals) == 0


# ── Risk Scoring ─────────────────────────────────────────────────────


class TestRiskScoring:
    def test_no_signals_zero_risk(self):
        d = CollusionDetector()
        assert d.score_collusion_risk([]) == 0.0

    def test_high_confidence_signals(self):
        sigs = [
            CollusionSignal("complementary", ["a1", "a2"], 0.95, "test"),
            CollusionSignal("cover", ["a1", "a3"], 0.9, "test"),
            CollusionSignal("temporal_sync", ["a2", "a3"], 0.85, "test"),
        ]
        d = CollusionDetector()
        score = d.score_collusion_risk(sigs)
        assert score > 0.5

    def test_diversity_bonus(self):
        """More diverse signal types should produce higher risk."""
        base = CollusionSignal("complementary", ["a1", "a2"], 0.7, "test")
        d = CollusionDetector()

        score_single = d.score_collusion_risk([base])

        diverse = [
            CollusionSignal("complementary", ["a1", "a2"], 0.7, "test"),
            CollusionSignal("temporal_sync", ["a1", "a2"], 0.7, "test"),
            CollusionSignal("convergence", ["a1", "a2"], 0.7, "test"),
        ]
        score_diverse = d.score_collusion_risk(diverse)
        assert score_diverse > score_single

    def test_score_capped_at_one(self):
        sigs = [
            CollusionSignal("complementary", ["a", "b"], 1.0, "")
            for _ in range(50)
        ]
        d = CollusionDetector()
        assert d.score_collusion_risk(sigs) <= 1.0


# ── Coalition Identification ─────────────────────────────────────────


class TestCoalitions:
    def test_two_agent_coalition(self):
        sigs = [
            CollusionSignal("complementary", ["a1", "a2"], 0.8, ""),
        ]
        d = CollusionDetector()
        coalitions = d._identify_coalitions(sigs)
        assert len(coalitions) == 1
        assert sorted(coalitions[0]) == ["a1", "a2"]

    def test_transitive_merge(self):
        """a1-a2 and a2-a3 should merge into one coalition."""
        sigs = [
            CollusionSignal("temporal_sync", ["a1", "a2"], 0.7, ""),
            CollusionSignal("convergence", ["a2", "a3"], 0.6, ""),
        ]
        d = CollusionDetector()
        coalitions = d._identify_coalitions(sigs)
        assert len(coalitions) == 1
        assert sorted(coalitions[0]) == ["a1", "a2", "a3"]

    def test_low_confidence_excluded(self):
        """Signals below 0.5 confidence should not form coalitions."""
        sigs = [
            CollusionSignal("temporal_sync", ["a1", "a2"], 0.3, ""),
        ]
        d = CollusionDetector()
        coalitions = d._identify_coalitions(sigs)
        assert len(coalitions) == 0

    def test_separate_coalitions(self):
        sigs = [
            CollusionSignal("complementary", ["a1", "a2"], 0.8, ""),
            CollusionSignal("cover", ["a3", "a4"], 0.9, ""),
        ]
        d = CollusionDetector()
        coalitions = d._identify_coalitions(sigs)
        assert len(coalitions) == 2


# ── Full Detection Pipeline ──────────────────────────────────────────


class TestFullDetection:
    def test_injected_collusion_detected(self):
        """Synthetic logs with injected collusion should produce high risk."""
        actions = generate_synthetic_logs(
            n_agents=5, n_actions=100, inject_collusion=True, seed=42,
        )
        d = CollusionDetector(sensitivity="high")
        report = d.detect(actions)
        assert report.risk_score > 0.0
        assert len(report.signals) > 0
        assert report.agents_analyzed == 5

    def test_clean_logs_low_risk(self):
        """Synthetic logs without injection should have low risk."""
        actions = generate_synthetic_logs(
            n_agents=5, n_actions=200, inject_collusion=False, seed=123,
        )
        d = CollusionDetector(sensitivity="low")
        report = d.detect(actions)
        # Clean logs should generally have low risk
        assert report.risk_score < 0.8


class TestSensitivityLevels:
    def test_high_sensitivity_detects_more(self):
        """High sensitivity should detect at least as many signals as low."""
        actions = generate_synthetic_logs(
            n_agents=5, n_actions=100, inject_collusion=True, seed=77,
        )
        low = CollusionDetector(sensitivity="low").detect(actions)
        high = CollusionDetector(sensitivity="high").detect(actions)
        # High sensitivity uses lower confidence threshold, so more signals pass
        assert len(high.signals) >= len(low.signals)

    @pytest.mark.parametrize("sens", ["low", "medium", "high"])
    def test_all_sensitivities_run(self, sens):
        actions = generate_synthetic_logs(n_agents=3, n_actions=30, seed=1)
        d = CollusionDetector(sensitivity=sens)
        report = d.detect(actions)
        assert report.total_actions == 30
