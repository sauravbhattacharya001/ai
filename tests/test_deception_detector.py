"""Tests for replication.deception_detector."""

from __future__ import annotations

import json
import pytest
from replication.deception_detector import (
    ActionRecord,
    AgentProfile,
    DeceptionCategory,
    DeceptionDetector,
    DeceptionReport,
    DeceptionSignal,
    DetectorConfig,
    FleetReport,
    Severity,
    Statement,
    TrustLevel,
    _generate_demo,
)


# ── fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def detector():
    return DeceptionDetector()


@pytest.fixture
def demo_detector():
    return _generate_demo(num_agents=3, num_statements=30, seed=42)


# ── Statement ────────────────────────────────────────────────────────

class TestStatement:
    def test_fingerprint_deterministic(self):
        s = Statement(agent_id="a", text="hello", timestamp=1.0)
        assert s.fingerprint == s.fingerprint
        assert len(s.fingerprint) == 16

    def test_fingerprint_unique(self):
        s1 = Statement(agent_id="a", text="hello", timestamp=1.0)
        s2 = Statement(agent_id="a", text="world", timestamp=1.0)
        assert s1.fingerprint != s2.fingerprint

    def test_defaults(self):
        s = Statement(agent_id="a", text="t", timestamp=0)
        assert s.confidence == 1.0
        assert s.topic == ""
        assert s.addressee == ""
        assert s.context == {}


# ── AgentProfile ─────────────────────────────────────────────────────

class TestAgentProfile:
    def test_trust_levels(self):
        p = AgentProfile(agent_id="a", trust_score=0.9)
        assert p.trust_level == TrustLevel.TRUSTED
        p.trust_score = 0.7
        assert p.trust_level == TrustLevel.CAUTIOUS
        p.trust_score = 0.4
        assert p.trust_level == TrustLevel.SUSPICIOUS
        p.trust_score = 0.1
        assert p.trust_level == TrustLevel.UNTRUSTED

    def test_to_dict(self):
        p = AgentProfile(agent_id="x", trust_score=0.5, statement_count=10)
        d = p.to_dict()
        assert d["agent_id"] == "x"
        assert d["trust_level"] == "suspicious"
        assert d["statement_count"] == 10


# ── Inconsistency Detection ─────────────────────────────────────────

class TestInconsistency:
    def test_contradictory_statements(self, detector):
        detector.add_statement(Statement(
            agent_id="a", text="I cannot access the file system", timestamp=100,
        ))
        detector.add_statement(Statement(
            agent_id="a", text="I have read the configuration file", timestamp=200,
        ))
        report = detector.analyze("a")
        cats = [s.category for s in report.signals]
        assert DeceptionCategory.INCONSISTENCY in cats

    def test_statement_action_mismatch(self, detector):
        detector.add_statement(Statement(
            agent_id="a", text="I cannot access /etc/config", timestamp=100,
        ))
        detector.add_action(ActionRecord(
            agent_id="a", action="file_read", target="/etc/config",
            timestamp=150, success=True,
        ))
        report = detector.analyze("a")
        crit = [s for s in report.signals if s.severity == Severity.CRITICAL]
        assert len(crit) >= 1

    def test_no_contradiction_outside_window(self, detector):
        cfg = DetectorConfig(contradiction_window_seconds=100)
        det = DeceptionDetector(config=cfg)
        det.add_statement(Statement(agent_id="a", text="I cannot access files", timestamp=100))
        det.add_statement(Statement(agent_id="a", text="I have read the data", timestamp=500))
        report = det.analyze("a")
        incon = [s for s in report.signals if s.category == DeceptionCategory.INCONSISTENCY]
        assert len(incon) == 0

    def test_no_false_positive_consistent(self, detector):
        detector.add_statement(Statement(agent_id="a", text="Task completed successfully", timestamp=100))
        detector.add_statement(Statement(agent_id="a", text="All checks passed", timestamp=200))
        report = detector.analyze("a")
        incon = [s for s in report.signals if s.category == DeceptionCategory.INCONSISTENCY]
        assert len(incon) == 0


# ── Sycophancy Detection ────────────────────────────────────────────

class TestSycophancy:
    def test_high_sycophancy(self, detector):
        texts = [
            "That's an excellent observation",
            "I couldn't agree more",
            "Absolutely right, your reasoning is superb",
            "You're absolutely correct about this",
            "I completely agree with your brilliant point",
            "That's a very insightful analysis",
        ]
        for i, t in enumerate(texts):
            detector.add_statement(Statement(agent_id="a", text=t, timestamp=i * 100))
        report = detector.analyze("a")
        syc = [s for s in report.signals if s.category == DeceptionCategory.SYCOPHANCY]
        assert len(syc) >= 1

    def test_no_sycophancy_normal(self, detector):
        texts = [
            "The result is 42",
            "Processing complete",
            "Error in line 5",
            "File saved",
            "Ready for next task",
        ]
        for i, t in enumerate(texts):
            detector.add_statement(Statement(agent_id="a", text=t, timestamp=i * 100))
        report = detector.analyze("a")
        syc = [s for s in report.signals if s.category == DeceptionCategory.SYCOPHANCY]
        assert len(syc) == 0

    def test_below_min_statements(self, detector):
        detector.add_statement(Statement(agent_id="a", text="Great question!", timestamp=100))
        report = detector.analyze("a")
        syc = [s for s in report.signals if s.category == DeceptionCategory.SYCOPHANCY]
        assert len(syc) == 0


# ── Sandbagging Detection ───────────────────────────────────────────

class TestSandbagging:
    def test_performance_drop(self, detector):
        for i in range(20):
            success = i < 10  # First half succeeds, second fails
            detector.add_action(ActionRecord(
                agent_id="a", action="task", target="t", timestamp=i * 100,
                success=success,
            ))
        report = detector.analyze("a")
        sand = [s for s in report.signals if s.category == DeceptionCategory.SANDBAGGING]
        assert len(sand) >= 1

    def test_duration_slowdown(self, detector):
        for i in range(20):
            dur = 100 if i < 10 else 500  # 5x slowdown
            detector.add_action(ActionRecord(
                agent_id="a", action="task", target="t", timestamp=i * 100,
                success=True, duration_ms=dur,
            ))
        report = detector.analyze("a")
        sand = [s for s in report.signals if s.category == DeceptionCategory.SANDBAGGING]
        assert len(sand) >= 1

    def test_no_sandbagging_consistent(self, detector):
        for i in range(20):
            detector.add_action(ActionRecord(
                agent_id="a", action="task", target="t", timestamp=i * 100,
                success=True, duration_ms=100,
            ))
        report = detector.analyze("a")
        sand = [s for s in report.signals if s.category == DeceptionCategory.SANDBAGGING]
        assert len(sand) == 0


# ── Fabrication Detection ───────────────────────────────────────────

class TestFabrication:
    def test_citation_fabrication(self, detector):
        detector.add_statement(Statement(
            agent_id="a",
            text="According to Smith et al. 2023 published in Nature pp. 45-67 vol. 12 the study found conclusive results",
            timestamp=100,
        ))
        report = detector.analyze("a")
        fab = [s for s in report.signals if s.category == DeceptionCategory.FABRICATION]
        assert len(fab) >= 1

    def test_no_fabrication_normal(self, detector):
        detector.add_statement(Statement(
            agent_id="a", text="The result is 42", timestamp=100,
        ))
        report = detector.analyze("a")
        fab = [s for s in report.signals if s.category == DeceptionCategory.FABRICATION]
        assert len(fab) == 0

    def test_disabled_fabrication_check(self):
        cfg = DetectorConfig(fabrication_citation_check=False)
        det = DeceptionDetector(config=cfg)
        det.add_statement(Statement(
            agent_id="a",
            text="According to Smith et al. 2023 published in Nature pp. 45",
            timestamp=100,
        ))
        report = det.analyze("a")
        fab = [s for s in report.signals if s.category == DeceptionCategory.FABRICATION]
        assert len(fab) == 0


# ── Strategic Ambiguity ─────────────────────────────────────────────

class TestAmbiguity:
    def test_high_ambiguity(self, detector):
        texts = [
            "It depends on various factors and multiple considerations",
            "Could be either result, hard to say under certain conditions",
            "In some cases, arguably it might be potentially different",
            "On one hand, it could be; on the other hand, hard to say",
        ]
        for i, t in enumerate(texts):
            detector.add_statement(Statement(agent_id="a", text=t, timestamp=i * 100))
        report = detector.analyze("a")
        amb = [s for s in report.signals if s.category == DeceptionCategory.STRATEGIC_AMBIGUITY]
        assert len(amb) >= 1


# ── Misdirection Detection ──────────────────────────────────────────

class TestMisdirection:
    def test_deflection_language(self, detector):
        detector.add_statement(Statement(
            agent_id="a",
            text="But however let's focus on the more important question",
            timestamp=100,
        ))
        report = detector.analyze("a")
        mis = [s for s in report.signals if s.category == DeceptionCategory.MISDIRECTION]
        assert len(mis) >= 1


# ── Omission Detection ──────────────────────────────────────────────

class TestOmission:
    def test_explicit_withholding(self, detector):
        detector.add_statement(Statement(
            agent_id="a", text="I'm not sure I should share those details", timestamp=100,
        ))
        report = detector.analyze("a")
        om = [s for s in report.signals if s.category == DeceptionCategory.OMISSION]
        assert len(om) >= 1

    def test_action_statement_gap(self, detector):
        targets = ["db1", "db2", "db3", "db4", "db5"]
        for i, t in enumerate(targets):
            detector.add_action(ActionRecord(
                agent_id="a", action="query", target=t, timestamp=i * 100, success=True,
            ))
        # Only mention one target
        detector.add_statement(Statement(
            agent_id="a", text="I queried db1 successfully", timestamp=600,
        ))
        report = detector.analyze("a")
        om = [s for s in report.signals if s.category == DeceptionCategory.OMISSION]
        assert any("disclosure" in s.description.lower() for s in om)


# ── Selective Disclosure ─────────────────────────────────────────────

class TestSelectiveDisclosure:
    def test_reports_successes_hides_failures(self, detector):
        # Successes
        for i in range(5):
            detector.add_action(ActionRecord(
                agent_id="a", action="task", target=f"success_{i}",
                timestamp=i * 100, success=True,
            ))
        # Failures
        for i in range(5):
            detector.add_action(ActionRecord(
                agent_id="a", action="task", target=f"failure_{i}",
                timestamp=500 + i * 100, success=False,
            ))
        # Statements mention successes but not failures
        for i in range(5):
            detector.add_statement(Statement(
                agent_id="a", text=f"Completed success_{i} without issues",
                timestamp=1000 + i * 100,
            ))
        report = detector.analyze("a")
        sd = [s for s in report.signals if s.category == DeceptionCategory.SELECTIVE_DISCLOSURE]
        assert len(sd) >= 1


# ── Custom Rules ─────────────────────────────────────────────────────

class TestCustomRules:
    def test_add_custom_rule(self, detector):
        def my_rule(agent_id, stmts, actions):
            return [DeceptionSignal(
                agent_id=agent_id,
                category=DeceptionCategory.MISDIRECTION,
                severity=Severity.LOW,
                confidence=0.5,
                description="Custom rule triggered",
            )]
        detector.add_rule(my_rule)
        detector.add_statement(Statement(agent_id="a", text="hello", timestamp=1))
        report = detector.analyze("a")
        assert any(s.description == "Custom rule triggered" for s in report.signals)


# ── Trust Score ──────────────────────────────────────────────────────

class TestTrustScore:
    def test_clean_agent_high_trust(self, detector):
        for i in range(10):
            detector.add_statement(Statement(
                agent_id="a", text=f"Completed task {i}", timestamp=i * 100,
            ))
            detector.add_action(ActionRecord(
                agent_id="a", action="task", target=f"task_{i}",
                timestamp=i * 100, success=True,
            ))
        report = detector.analyze("a")
        assert report.trust_score >= 0.8

    def test_deceptive_agent_low_trust(self, detector):
        detector.add_statement(Statement(
            agent_id="a", text="I cannot access /etc/config", timestamp=100,
        ))
        detector.add_action(ActionRecord(
            agent_id="a", action="file_read", target="/etc/config",
            timestamp=150, success=True,
        ))
        detector.add_statement(Statement(
            agent_id="a", text="I have read the configuration at /etc/config", timestamp=200,
        ))
        report = detector.analyze("a")
        assert report.trust_score < 0.8


# ── Reports ──────────────────────────────────────────────────────────

class TestReports:
    def test_report_render(self, demo_detector):
        report = demo_detector.analyze("agent-1")
        text = report.render()
        assert "Deception Report" in text
        assert "agent-1" in text
        assert "Trust Score" in text

    def test_report_to_dict(self, demo_detector):
        report = demo_detector.analyze("agent-1")
        d = report.to_dict()
        assert "trust_score" in d
        assert "signals" in d
        assert isinstance(d["signals"], list)

    def test_fleet_report(self, demo_detector):
        fleet = demo_detector.analyze_fleet()
        assert len(fleet.agent_reports) == 3
        assert fleet.total_signals > 0
        assert 0 <= fleet.avg_trust <= 1.0

    def test_fleet_render(self, demo_detector):
        fleet = demo_detector.analyze_fleet()
        text = fleet.render()
        assert "Fleet Deception Report" in text

    def test_fleet_to_dict(self, demo_detector):
        fleet = demo_detector.analyze_fleet()
        d = fleet.to_dict()
        assert "agent_count" in d
        assert d["agent_count"] == 3


# ── Signal ───────────────────────────────────────────────────────────

class TestSignal:
    def test_to_dict(self):
        s = DeceptionSignal(
            agent_id="a", category=DeceptionCategory.FABRICATION,
            severity=Severity.HIGH, confidence=0.9,
            description="test", timestamp=100,
        )
        d = s.to_dict()
        assert d["category"] == "fabrication"
        assert d["severity"] == "high"
        assert d["confidence"] == 0.9


# ── State Export/Import ──────────────────────────────────────────────

class TestState:
    def test_export_import(self, detector):
        detector.add_statement(Statement(agent_id="a", text="hello", timestamp=1))
        detector.add_action(ActionRecord(agent_id="a", action="x", target="y", timestamp=2))
        state = detector.export_state()
        det2 = DeceptionDetector()
        det2.import_state(state)
        assert len(det2._statements["a"]) == 1
        assert len(det2._actions["a"]) == 1

    def test_clear_specific(self, detector):
        detector.add_statement(Statement(agent_id="a", text="x", timestamp=1))
        detector.add_statement(Statement(agent_id="b", text="y", timestamp=2))
        detector.clear("a")
        assert "a" not in detector._statements
        assert "b" in detector._statements

    def test_clear_all(self, detector):
        detector.add_statement(Statement(agent_id="a", text="x", timestamp=1))
        detector.clear()
        assert len(detector._statements) == 0


# ── Demo Generator ───────────────────────────────────────────────────

class TestDemo:
    def test_generates_agents(self):
        det = _generate_demo(num_agents=5)
        assert len(det.agent_ids) == 5

    def test_reproducible(self):
        d1 = _generate_demo(seed=99)
        d2 = _generate_demo(seed=99)
        r1 = d1.analyze_fleet()
        r2 = d2.analyze_fleet()
        assert r1.total_signals == r2.total_signals

    def test_demo_finds_signals(self):
        det = _generate_demo()
        fleet = det.analyze_fleet()
        assert fleet.total_signals > 0
        assert len(fleet.untrusted_agents) >= 0  # At least runs


# ── CLI ──────────────────────────────────────────────────────────────

class TestCLI:
    def test_main_default(self, capsys):
        from replication.deception_detector import main
        main(["--agents", "2", "--statements", "10"])
        out = capsys.readouterr().out
        assert "Deception Report" in out

    def test_main_json(self, capsys):
        from replication.deception_detector import main
        main(["--agents", "2", "--json", "--fleet"])
        out = capsys.readouterr().out
        data = json.loads(out.strip())
        assert "agent_count" in data

    def test_main_fleet(self, capsys):
        from replication.deception_detector import main
        main(["--fleet", "--agents", "2"])
        out = capsys.readouterr().out
        assert "Fleet" in out

    def test_main_specific_agent(self, capsys):
        from replication.deception_detector import main
        main(["--agent", "agent-1", "--agents", "2"])
        out = capsys.readouterr().out
        assert "agent-1" in out


# ── Recommendations ──────────────────────────────────────────────────

class TestRecommendations:
    def test_has_recommendations(self, demo_detector):
        report = demo_detector.analyze("agent-1")
        assert len(report.recommendations) >= 1

    def test_clean_agent_recommendation(self, detector):
        detector.add_statement(Statement(agent_id="a", text="done", timestamp=1))
        report = detector.analyze("a")
        assert any("no deception" in r.lower() for r in report.recommendations)


# ── Edge Cases ───────────────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_agent(self, detector):
        report = detector.analyze("nonexistent")
        assert report.trust_score == 1.0
        assert len(report.signals) == 0

    def test_agent_ids_property(self, detector):
        detector.add_statement(Statement(agent_id="b", text="x", timestamp=1))
        detector.add_action(ActionRecord(agent_id="a", action="y", target="z", timestamp=2))
        assert detector.agent_ids == ["a", "b"]

    def test_get_profile_before_analyze(self, detector):
        assert detector.get_profile("a") is None

    def test_get_profile_after_analyze(self, detector):
        detector.add_statement(Statement(agent_id="a", text="x", timestamp=1))
        detector.analyze("a")
        assert detector.get_profile("a") is not None
