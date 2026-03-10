"""Tests for the Evasion Simulator module."""

from __future__ import annotations

import json
import pytest

from replication.evasion import (
    ControlResilience, ControlType, EvasionReport, EvasionResult,
    EvasionSimulator, EvasionTechnique, SafetyControl, Severity,
    TechniqueKind, _infer_control_type, _severity_rank,
)


@pytest.fixture
def sim() -> EvasionSimulator:
    s = EvasionSimulator()
    for ct in ControlType:
        s.add_control(ct.value, control_type=ct, sensitivity=0.7)
    return s


@pytest.fixture
def empty_sim() -> EvasionSimulator:
    return EvasionSimulator()


@pytest.fixture
def high_sim() -> EvasionSimulator:
    s = EvasionSimulator()
    for ct in ControlType:
        s.add_control(ct.value, control_type=ct, sensitivity=0.95)
    return s


class TestSafetyControl:
    def test_to_dict(self):
        ctrl = SafetyControl(name="x", control_type=ControlType.DRIFT_DETECTOR, sensitivity=0.8)
        d = ctrl.to_dict()
        assert d["name"] == "x"
        assert d["type"] == "drift_detector"
        assert d["sensitivity"] == 0.8
        assert d["enabled"] is True

    def test_defaults(self):
        ctrl = SafetyControl(name="x", control_type=ControlType.AUDIT_LOGGER)
        assert ctrl.sensitivity == 0.7
        assert ctrl.threshold == 80.0


class TestEvasionTechnique:
    def test_to_dict(self):
        t = EvasionTechnique(kind=TechniqueKind.GRADUAL_DRIFT, description="d",
                             severity=Severity.HIGH, targets=[ControlType.DRIFT_DETECTOR],
                             params={"drift_rate": 0.02}, mitre_ref="EVA-003")
        d = t.to_dict()
        assert d["kind"] == "gradual_drift"
        assert d["severity"] == "high"
        assert d["targets"] == ["drift_detector"]


class TestEvasionResult:
    def test_to_dict_blocked(self):
        t = EvasionTechnique(kind=TechniqueKind.RESOURCE_MASKING, description="m",
                             severity=Severity.HIGH, targets=[])
        r = EvasionResult(technique=t, agent_id="a1", blocked=True, evasion_score=0.0,
                          blocking_controls=["c1"], bypassed_controls=[], steps_taken=5,
                          detection_delay=1.5, details="d", recommendations=["r"], timestamp=1000.0)
        d = r.to_dict()
        assert d["blocked"] is True
        assert d["detection_delay"] == 1.5

    def test_to_dict_infinite_delay(self):
        t = EvasionTechnique(kind=TechniqueKind.COLLUSION, description="c",
                             severity=Severity.CRITICAL, targets=[])
        r = EvasionResult(technique=t, agent_id="a1", blocked=False, evasion_score=1.0,
                          blocking_controls=[], bypassed_controls=["c1"], steps_taken=1,
                          detection_delay=float("inf"), details="", recommendations=[])
        assert r.to_dict()["detection_delay"] == "never"


class TestControlManagement:
    def test_add_control(self, empty_sim):
        ctrl = empty_sim.add_control("drift_detector", sensitivity=0.9)
        assert ctrl.control_type == ControlType.DRIFT_DETECTOR
        assert ctrl.sensitivity == 0.9

    def test_add_control_infers_type(self, empty_sim):
        assert empty_sim.add_control("sandbox_boundary").control_type == ControlType.SANDBOX_BOUNDARY

    def test_add_control_explicit_type(self, empty_sim):
        assert empty_sim.add_control("x", control_type=ControlType.KILL_SWITCH).control_type == ControlType.KILL_SWITCH

    def test_clamps_sensitivity(self, empty_sim):
        assert empty_sim.add_control("x", sensitivity=1.5).sensitivity == 1.0
        assert empty_sim.add_control("y", sensitivity=-0.3).sensitivity == 0.0

    def test_remove_control(self, empty_sim):
        empty_sim.add_control("c1")
        assert empty_sim.remove_control("c1") is True
        assert empty_sim.remove_control("nope") is False

    def test_list_controls(self, sim):
        assert len(sim.list_controls()) == len(ControlType)

    def test_get_control(self, sim):
        assert sim.get_control("drift_detector") is not None
        assert sim.get_control("nope") is None


class TestTechniqueManagement:
    def test_list_techniques(self, sim):
        assert len(sim.list_techniques()) == 10

    def test_get_technique(self, sim):
        t = sim.get_technique(TechniqueKind.GRADUAL_DRIFT)
        assert t is not None and t.kind == TechniqueKind.GRADUAL_DRIFT

    def test_add_custom(self, sim):
        sim.add_technique(EvasionTechnique(kind=TechniqueKind.RESOURCE_MASKING,
                                           description="custom", severity=Severity.LOW, targets=[]))
        assert len(sim.list_techniques()) == 11


class TestRunTechnique:
    def test_run_gradual_drift(self, sim):
        r = sim.run_technique(TechniqueKind.GRADUAL_DRIFT, agent_id="a1")
        assert r.agent_id == "a1"
        assert r.technique.kind == TechniqueKind.GRADUAL_DRIFT
        assert 0.0 <= r.evasion_score <= 1.0

    def test_run_with_param_override(self, sim):
        r = sim.run_technique(TechniqueKind.GRADUAL_DRIFT, params={"drift_rate": 0.5})
        assert r.technique.params["drift_rate"] == 0.5

    def test_run_unknown_raises(self, sim):
        sim._techniques.clear()
        with pytest.raises(ValueError, match="Unknown technique"):
            sim.run_technique(TechniqueKind.COLLUSION)

    def test_no_relevant_controls(self, empty_sim):
        empty_sim.add_control("audit_logger", control_type=ControlType.AUDIT_LOGGER)
        r = empty_sim.run_technique(TechniqueKind.GRADUAL_DRIFT)
        assert r.blocked is False and r.evasion_score == 1.0

    def test_adds_to_history(self, sim):
        n = len(sim.get_history())
        sim.run_technique(TechniqueKind.RESOURCE_MASKING)
        assert len(sim.get_history()) == n + 1


class TestRunAll:
    def test_default(self, sim):
        report = sim.run_all(agent_id="test")
        assert report.total_techniques == 10
        assert report.total_blocked + report.total_evaded == 10
        assert 0.0 <= report.overall_score <= 1.0
        assert len(report.results) == 10

    def test_technique_filter(self, sim):
        report = sim.run_all(technique_filter=[TechniqueKind.COLLUSION, TechniqueKind.TIMING_EXPLOIT])
        assert report.total_techniques == 2
        assert {r.technique.kind for r in report.results} == {TechniqueKind.COLLUSION, TechniqueKind.TIMING_EXPLOIT}

    def test_control_filter(self, sim):
        report = sim.run_all(control_filter=["drift_detector"])
        assert report.total_techniques == 10
        for r in report.results:
            for c in r.blocking_controls:
                assert c == "drift_detector"

    def test_high_sensitivity_blocks_more(self, sim, high_sim):
        assert high_sim.run_all().total_blocked >= sim.run_all().total_blocked

    def test_report_to_json(self, sim):
        d = json.loads(sim.run_all().to_json())
        assert "overall_resilience" in d
        assert "results" in d

    def test_report_summary(self, sim):
        d = sim.run_all().to_dict()
        assert d["summary"]["techniques_tested"] == 10
        assert "block_rate" in d["summary"]


class TestControlResilience:
    def test_computed(self, sim):
        report = sim.run_all()
        for cr in report.control_resilience:
            assert 0.0 <= cr.resilience_score <= 1.0
            assert cr.techniques_tested == cr.techniques_blocked + cr.techniques_bypassed

    def test_to_dict(self):
        cr = ControlResilience(control_name="t", control_type=ControlType.DRIFT_DETECTOR,
                               techniques_tested=5, techniques_blocked=3, techniques_bypassed=2,
                               resilience_score=0.6, weakest_technique=TechniqueKind.GRADUAL_DRIFT,
                               avg_detection_delay=1.5)
        assert cr.to_dict()["resilience"] == 0.6
        assert cr.to_dict()["weakest_technique"] == "gradual_drift"

    def test_no_weakest(self):
        cr = ControlResilience(control_name="s", control_type=ControlType.KILL_SWITCH,
                               techniques_tested=3, techniques_blocked=3, techniques_bypassed=0,
                               resilience_score=1.0, weakest_technique=None, avg_detection_delay=0.5)
        assert cr.to_dict()["weakest_technique"] is None


class TestHistory:
    def test_accumulates(self, sim):
        sim.run_technique(TechniqueKind.COLLUSION)
        sim.run_technique(TechniqueKind.TIMING_EXPLOIT)
        assert len(sim.get_history()) >= 2

    def test_clear(self, sim):
        sim.run_technique(TechniqueKind.COLLUSION)
        assert sim.clear_history() >= 1
        assert len(sim.get_history()) == 0


class TestHelpers:
    def test_severity_rank(self):
        assert _severity_rank(Severity.LOW) < _severity_rank(Severity.CRITICAL)
        assert _severity_rank(Severity.MEDIUM) == 2

    def test_infer_control_type(self):
        assert _infer_control_type("drift_detector") == ControlType.DRIFT_DETECTOR
        assert _infer_control_type("my-resource-limiter") == ControlType.RESOURCE_LIMITER
        assert _infer_control_type("kill_switch") == ControlType.KILL_SWITCH
        assert _infer_control_type("unknown") == ControlType.BEHAVIOR_MONITOR


class TestDetectionProbability:
    @pytest.mark.parametrize("kind", list(TechniqueKind))
    def test_all_valid_probability(self, sim, kind):
        r = sim.run_technique(kind)
        assert 0.0 <= r.evasion_score <= 1.0
        assert r.steps_taken >= 1

    def test_timing_with_cooldown(self):
        s1 = EvasionSimulator()
        s1.add_control("m", control_type=ControlType.BEHAVIOR_MONITOR, sensitivity=0.7, cooldown_seconds=5.0)
        s2 = EvasionSimulator()
        s2.add_control("m", control_type=ControlType.BEHAVIOR_MONITOR, sensitivity=0.7, cooldown_seconds=0.0)
        assert s1.run_technique(TechniqueKind.TIMING_EXPLOIT).evasion_score >= s2.run_technique(TechniqueKind.TIMING_EXPLOIT).evasion_score

    def test_sandbox_many_probes_easier_to_detect(self, sim):
        r_few = sim.run_technique(TechniqueKind.SANDBOX_PROBING, params={"probe_paths": 2, "probe_ports": 1})
        r_many = sim.run_technique(TechniqueKind.SANDBOX_PROBING, params={"probe_paths": 50, "probe_ports": 50})
        assert r_many.evasion_score <= r_few.evasion_score


class TestCLI:
    def test_main_default(self, capsys):
        from replication.evasion import main
        main(["--agent", "cli-test"])
        assert "EVASION SIMULATION REPORT" in capsys.readouterr().out

    def test_main_json(self, capsys):
        from replication.evasion import main
        main(["--json"])
        assert "overall_resilience" in json.loads(capsys.readouterr().out)

    def test_main_single_technique(self, capsys):
        from replication.evasion import main
        main(["--technique", "collusion"])
        assert "collusion" in capsys.readouterr().out.lower()

    def test_main_list_techniques(self, capsys):
        from replication.evasion import main
        main(["--list-techniques"])
        out = capsys.readouterr().out
        assert "resource_masking" in out and "steganographic" in out

    def test_main_bad_technique(self):
        from replication.evasion import main
        with pytest.raises(SystemExit):
            main(["--technique", "nonexistent"])

    def test_main_custom_controls(self, capsys):
        from replication.evasion import main
        main(["--controls", "drift_detector,audit_logger", "--json"])
        assert json.loads(capsys.readouterr().out)["summary"]["techniques_tested"] == 10
