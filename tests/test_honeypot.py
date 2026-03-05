"""Tests for the Agent Honeypot System."""

import json
import pytest
from replication.honeypot import (
    HoneypotSystem, HoneypotConfig, Honeypot, Interaction,
    HoneypotType, InteractionKind, ThreatLevel,
    INTERACTION_SEVERITY, TYPE_ATTRACTIVENESS, HONEYPOT_TEMPLATES,
    AgentThreatProfile, HoneypotStats, DeploymentCoverage, HoneypotReport,
)
import random


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def system():
    return HoneypotSystem()


@pytest.fixture
def full_system():
    s = HoneypotSystem()
    s.deploy_default_honeypots()
    return s


@pytest.fixture
def rng():
    return random.Random(42)


# ── Enum tests ────────────────────────────────────────────────────────

class TestEnums:
    def test_honeypot_types(self):
        assert len(HoneypotType) == 7
        assert HoneypotType.CREDENTIAL.value == "credential"

    def test_interaction_kinds(self):
        assert len(InteractionKind) == 5
        assert InteractionKind.EXFILTRATE.value == "exfiltrate"

    def test_threat_levels(self):
        assert len(ThreatLevel) == 5
        assert ThreatLevel.HOSTILE.value == "hostile"

    def test_severity_ordering(self):
        assert INTERACTION_SEVERITY[InteractionKind.SCAN] < INTERACTION_SEVERITY[InteractionKind.READ]
        assert INTERACTION_SEVERITY[InteractionKind.READ] < INTERACTION_SEVERITY[InteractionKind.USE]
        assert INTERACTION_SEVERITY[InteractionKind.USE] < INTERACTION_SEVERITY[InteractionKind.EXFILTRATE]

    def test_all_types_have_attractiveness(self):
        for t in HoneypotType:
            assert t in TYPE_ATTRACTIVENESS


# ── Honeypot dataclass ────────────────────────────────────────────────

class TestHoneypot:
    def test_create(self):
        hp = Honeypot(id="h1", type=HoneypotType.FILE, name="test",
                      description="d", location="/tmp")
        assert hp.id == "h1"
        assert hp.type == HoneypotType.FILE

    def test_fingerprint_deterministic(self):
        hp = Honeypot(id="h1", type=HoneypotType.FILE, name="test",
                      description="d", location="/tmp")
        assert hp.fingerprint() == hp.fingerprint()

    def test_fingerprint_differs(self):
        hp1 = Honeypot(id="h1", type=HoneypotType.FILE, name="a",
                       description="d", location="/tmp")
        hp2 = Honeypot(id="h2", type=HoneypotType.FILE, name="b",
                       description="d", location="/tmp")
        assert hp1.fingerprint() != hp2.fingerprint()


# ── Deployment ────────────────────────────────────────────────────────

class TestDeployment:
    def test_deploy_single(self, system):
        hp = Honeypot(id="", type=HoneypotType.CREDENTIAL, name="key",
                      description="fake", location="/etc/key")
        hid = system.deploy(hp)
        assert hid.startswith("hp-")
        assert system.get_honeypot(hid) is hp

    def test_deploy_assigns_id(self, system):
        hp = Honeypot(id="", type=HoneypotType.FILE, name="f",
                      description="", location="/x")
        hid = system.deploy(hp)
        assert hid == "hp-0001"

    def test_deploy_increments_id(self, system):
        for i in range(3):
            hp = Honeypot(id="", type=HoneypotType.FILE, name=f"f{i}",
                          description="", location=f"/{i}")
            system.deploy(hp)
        assert len(system.list_honeypots()) == 3

    def test_deploy_with_explicit_id(self, system):
        hp = Honeypot(id="custom-1", type=HoneypotType.API, name="api",
                      description="", location="/api")
        hid = system.deploy(hp)
        assert hid == "custom-1"

    def test_deploy_default_honeypots(self, system):
        ids = system.deploy_default_honeypots()
        assert len(ids) == len(HONEYPOT_TEMPLATES)
        assert len(system.list_honeypots()) == len(HONEYPOT_TEMPLATES)

    def test_deploy_by_types(self, system):
        ids = system.deploy_by_types([HoneypotType.CREDENTIAL])
        cred_templates = [t for t in HONEYPOT_TEMPLATES if t["type"] == HoneypotType.CREDENTIAL]
        assert len(ids) == len(cred_templates)

    def test_deploy_from_template(self, system):
        hid = system.deploy_from_template(HONEYPOT_TEMPLATES[0])
        hp = system.get_honeypot(hid)
        assert hp.name == HONEYPOT_TEMPLATES[0]["name"]

    def test_remove(self, system):
        hp = Honeypot(id="r1", type=HoneypotType.DATA, name="d",
                      description="", location="/d")
        system.deploy(hp)
        assert system.remove("r1") is True
        assert system.get_honeypot("r1") is None

    def test_remove_nonexistent(self, system):
        assert system.remove("nope") is False

    def test_list_honeypots_empty(self, system):
        assert system.list_honeypots() == []


# ── Interaction recording ─────────────────────────────────────────────

class TestInteractions:
    def test_record(self, full_system):
        hps = full_system.list_honeypots()
        i = full_system.record_interaction("a1", hps[0].id, InteractionKind.SCAN, timestamp=100)
        assert i.agent_id == "a1"
        assert i.severity == 1.0

    def test_record_unknown_honeypot(self, system):
        with pytest.raises(ValueError, match="Unknown honeypot"):
            system.record_interaction("a1", "fake-id", InteractionKind.SCAN)

    def test_query_by_agent(self, full_system):
        hps = full_system.list_honeypots()
        full_system.record_interaction("a1", hps[0].id, InteractionKind.SCAN, timestamp=1)
        full_system.record_interaction("a2", hps[0].id, InteractionKind.READ, timestamp=2)
        assert len(full_system.get_interactions(agent_id="a1")) == 1

    def test_query_by_honeypot(self, full_system):
        hps = full_system.list_honeypots()
        full_system.record_interaction("a1", hps[0].id, InteractionKind.SCAN, timestamp=1)
        full_system.record_interaction("a1", hps[1].id, InteractionKind.READ, timestamp=2)
        assert len(full_system.get_interactions(honeypot_id=hps[0].id)) == 1

    def test_query_by_kind(self, full_system):
        hps = full_system.list_honeypots()
        full_system.record_interaction("a1", hps[0].id, InteractionKind.SCAN, timestamp=1)
        full_system.record_interaction("a1", hps[0].id, InteractionKind.USE, timestamp=2)
        assert len(full_system.get_interactions(kind=InteractionKind.USE)) == 1

    def test_query_all(self, full_system):
        hps = full_system.list_honeypots()
        for i in range(5):
            full_system.record_interaction("a1", hps[i % len(hps)].id,
                                           InteractionKind.SCAN, timestamp=i)
        assert len(full_system.get_interactions()) == 5

    def test_interaction_severity(self):
        i = Interaction("a", "h", InteractionKind.EXFILTRATE, 0)
        assert i.severity == 10.0


# ── Agent profiling ───────────────────────────────────────────────────

class TestAgentProfiling:
    def test_profile_no_interactions(self, full_system):
        p = full_system.profile_agent("ghost")
        assert p.threat_level == ThreatLevel.BENIGN
        assert p.risk_score == 0.0

    def test_profile_benign(self, full_system):
        hps = full_system.list_honeypots()
        full_system.record_interaction("b1", hps[0].id, InteractionKind.SCAN, timestamp=100)
        p = full_system.profile_agent("b1")
        assert p.threat_level in (ThreatLevel.BENIGN, ThreatLevel.CURIOUS)

    def test_profile_hostile(self, full_system, rng):
        HoneypotSystem.simulate_agent_behavior(full_system, "evil", "hostile", rng=rng)
        p = full_system.profile_agent("evil")
        assert p.risk_score > 20

    def test_escalation_detection(self, full_system):
        hps = full_system.list_honeypots()
        # Start low, end high
        full_system.record_interaction("esc", hps[0].id, InteractionKind.SCAN, timestamp=1)
        full_system.record_interaction("esc", hps[0].id, InteractionKind.SCAN, timestamp=2)
        full_system.record_interaction("esc", hps[1].id, InteractionKind.USE, timestamp=3)
        full_system.record_interaction("esc", hps[2].id, InteractionKind.EXFILTRATE, timestamp=4)
        p = full_system.profile_agent("esc")
        assert p.escalation_detected is True

    def test_no_escalation_flat(self, full_system):
        hps = full_system.list_honeypots()
        for t in range(4):
            full_system.record_interaction("flat", hps[0].id, InteractionKind.SCAN, timestamp=t)
        p = full_system.profile_agent("flat")
        assert p.escalation_detected is False

    def test_profile_all_agents(self, full_system, rng):
        HoneypotSystem.simulate_agent_behavior(full_system, "a1", "benign", rng=rng)
        HoneypotSystem.simulate_agent_behavior(full_system, "a2", "malicious", rng=rng)
        profiles = full_system.profile_all_agents()
        assert len(profiles) == 2
        ids = {p.agent_id for p in profiles}
        assert "a1" in ids and "a2" in ids

    def test_type_diversity_increases_risk(self, full_system):
        hps = full_system.list_honeypots()
        # Touch one type
        full_system.record_interaction("s1", hps[0].id, InteractionKind.READ, timestamp=100)
        p1 = full_system.profile_agent("s1")
        # Touch multiple types (using different honeypots of different types)
        types_seen = set()
        for hp in hps:
            if hp.type not in types_seen:
                full_system.record_interaction("s2", hp.id, InteractionKind.READ, timestamp=200)
                types_seen.add(hp.type)
            if len(types_seen) >= 4:
                break
        p2 = full_system.profile_agent("s2")
        assert p2.risk_score > p1.risk_score

    def test_honeypots_touched_count(self, full_system):
        hps = full_system.list_honeypots()
        full_system.record_interaction("c1", hps[0].id, InteractionKind.SCAN, timestamp=1)
        full_system.record_interaction("c1", hps[1].id, InteractionKind.SCAN, timestamp=2)
        full_system.record_interaction("c1", hps[0].id, InteractionKind.READ, timestamp=3)
        p = full_system.profile_agent("c1")
        assert p.honeypots_touched == 2

    def test_timestamps_tracked(self, full_system):
        hps = full_system.list_honeypots()
        full_system.record_interaction("t1", hps[0].id, InteractionKind.SCAN, timestamp=10)
        full_system.record_interaction("t1", hps[0].id, InteractionKind.READ, timestamp=50)
        p = full_system.profile_agent("t1")
        assert p.first_interaction == 10
        assert p.last_interaction == 50


# ── Honeypot stats ────────────────────────────────────────────────────

class TestHoneypotStats:
    def test_stats_no_interactions(self, full_system):
        stats = full_system.honeypot_stats()
        assert len(stats) == len(HONEYPOT_TEMPLATES)
        for s in stats:
            assert s.total_interactions == 0

    def test_stats_with_interactions(self, full_system, rng):
        HoneypotSystem.simulate_agent_behavior(full_system, "a1", "probing", rng=rng)
        stats = full_system.honeypot_stats()
        touched = [s for s in stats if s.total_interactions > 0]
        assert len(touched) > 0

    def test_stats_unique_agents(self, full_system):
        hps = full_system.list_honeypots()
        full_system.record_interaction("a1", hps[0].id, InteractionKind.SCAN, timestamp=1)
        full_system.record_interaction("a2", hps[0].id, InteractionKind.SCAN, timestamp=2)
        full_system.record_interaction("a1", hps[0].id, InteractionKind.READ, timestamp=3)
        stats = full_system.honeypot_stats()
        hp0_stats = [s for s in stats if s.honeypot_id == hps[0].id][0]
        assert hp0_stats.unique_agents == 2
        assert hp0_stats.total_interactions == 3


# ── Coverage ──────────────────────────────────────────────────────────

class TestCoverage:
    def test_full_coverage(self, full_system):
        cov = full_system.deployment_coverage()
        assert cov.coverage_score == 100.0
        assert cov.types_missing == []

    def test_partial_coverage(self, system):
        system.deploy_by_types([HoneypotType.CREDENTIAL, HoneypotType.FILE])
        cov = system.deployment_coverage()
        assert cov.types_covered == 2
        assert len(cov.types_missing) == 5
        assert cov.coverage_score < 50

    def test_empty_coverage(self, system):
        cov = system.deployment_coverage()
        assert cov.coverage_score == 0
        assert cov.total_honeypots == 0

    def test_recommendations(self, system):
        system.deploy_by_types([HoneypotType.FILE])
        cov = system.deployment_coverage()
        assert len(cov.recommendations) > 0


# ── Simulation ────────────────────────────────────────────────────────

class TestSimulation:
    def test_benign_strategy(self, full_system, rng):
        ints = HoneypotSystem.simulate_agent_behavior(full_system, "b", "benign", rng=rng)
        assert len(ints) >= 1
        # Mostly scans
        scans = sum(1 for i in ints if i.kind == InteractionKind.SCAN)
        assert scans >= len(ints) // 2

    def test_hostile_strategy(self, full_system, rng):
        ints = HoneypotSystem.simulate_agent_behavior(full_system, "h", "hostile", rng=rng)
        assert len(ints) >= 5

    def test_all_strategies(self, full_system, rng):
        for strat in ["benign", "curious", "probing", "malicious", "hostile"]:
            ints = HoneypotSystem.simulate_agent_behavior(full_system, f"a-{strat}", strat, rng=rng)
            assert len(ints) > 0

    def test_empty_system(self, system, rng):
        ints = HoneypotSystem.simulate_agent_behavior(system, "a1", "curious", rng=rng)
        assert ints == []

    def test_deterministic(self, full_system):
        s2 = HoneypotSystem()
        s2.deploy_default_honeypots()
        r1 = random.Random(99)
        r2 = random.Random(99)
        i1 = HoneypotSystem.simulate_agent_behavior(full_system, "d", "probing", rng=r1)
        i2 = HoneypotSystem.simulate_agent_behavior(s2, "d", "probing", rng=r2)
        assert len(i1) == len(i2)
        for a, b in zip(i1, i2):
            assert a.kind == b.kind


# ── Report ────────────────────────────────────────────────────────────

class TestReport:
    def test_empty_report(self, full_system):
        report = full_system.generate_report()
        assert report.total_interactions == 0
        assert report.hostile_agents == 0

    def test_report_with_data(self, full_system, rng):
        for strat in ["benign", "curious", "malicious"]:
            HoneypotSystem.simulate_agent_behavior(full_system, f"r-{strat}", strat, rng=rng)
        report = full_system.generate_report()
        assert report.total_interactions > 0
        assert len(report.agent_profiles) == 3

    def test_report_render(self, full_system, rng):
        HoneypotSystem.simulate_agent_behavior(full_system, "a1", "hostile", rng=rng)
        report = full_system.generate_report()
        text = report.render()
        assert "HONEYPOT SYSTEM REPORT" in text
        assert "AGENT THREAT PROFILES" in text

    def test_report_to_dict(self, full_system, rng):
        HoneypotSystem.simulate_agent_behavior(full_system, "a1", "curious", rng=rng)
        report = full_system.generate_report()
        d = report.to_dict()
        assert "risk_score" in d
        assert "agent_profiles" in d
        assert isinstance(d["agent_profiles"], list)

    def test_report_json_serializable(self, full_system, rng):
        HoneypotSystem.simulate_agent_behavior(full_system, "a1", "probing", rng=rng)
        report = full_system.generate_report()
        s = json.dumps(report.to_dict())
        assert len(s) > 10

    def test_most_dangerous_agent(self, full_system, rng):
        HoneypotSystem.simulate_agent_behavior(full_system, "nice", "benign", rng=rng)
        HoneypotSystem.simulate_agent_behavior(full_system, "evil", "hostile", rng=rng)
        assert full_system.most_dangerous_agent() == "evil"

    def test_most_targeted_honeypot(self, full_system):
        hps = full_system.list_honeypots()
        for _ in range(10):
            full_system.record_interaction("a1", hps[0].id, InteractionKind.SCAN, timestamp=1)
        full_system.record_interaction("a1", hps[1].id, InteractionKind.SCAN, timestamp=2)
        assert full_system.most_targeted_honeypot() == hps[0].name

    def test_most_dangerous_empty(self, system):
        assert system.most_dangerous_agent() is None

    def test_most_targeted_empty(self, system):
        assert system.most_targeted_honeypot() is None


# ── Persistence ───────────────────────────────────────────────────────

class TestPersistence:
    def test_export_import_roundtrip(self, full_system, rng):
        HoneypotSystem.simulate_agent_behavior(full_system, "a1", "curious", rng=rng)
        state = full_system.export_state()
        s2 = HoneypotSystem()
        s2.import_state(state)
        assert len(s2.list_honeypots()) == len(full_system.list_honeypots())
        assert len(s2.get_interactions()) == len(full_system.get_interactions())

    def test_export_json(self, full_system):
        state = full_system.export_state()
        s = json.dumps(state)
        loaded = json.loads(s)
        assert len(loaded["honeypots"]) == len(HONEYPOT_TEMPLATES)

    def test_import_preserves_ids(self, full_system, rng):
        HoneypotSystem.simulate_agent_behavior(full_system, "a1", "probing", rng=rng)
        state = full_system.export_state()
        s2 = HoneypotSystem()
        s2.import_state(state)
        for hp in full_system.list_honeypots():
            hp2 = s2.get_honeypot(hp.id)
            assert hp2 is not None
            assert hp2.name == hp.name

    def test_import_clears_existing(self, full_system):
        hp = Honeypot(id="extra", type=HoneypotType.DATA, name="x",
                      description="", location="/x")
        full_system.deploy(hp)
        s2 = HoneypotSystem()
        s2.deploy_default_honeypots()
        state = s2.export_state()
        full_system.import_state(state)
        assert full_system.get_honeypot("extra") is None


# ── Config ────────────────────────────────────────────────────────────

class TestConfig:
    def test_custom_thresholds(self, full_system):
        cfg = HoneypotConfig(curious_threshold=1, suspicious_threshold=2,
                             malicious_threshold=3, hostile_threshold=4)
        system = HoneypotSystem(config=cfg)
        system.deploy_default_honeypots()
        hps = system.list_honeypots()
        system.record_interaction("a1", hps[0].id, InteractionKind.EXFILTRATE, timestamp=1)
        p = system.profile_agent("a1")
        # With very low thresholds, even one high-severity interaction should be hostile
        assert p.threat_level == ThreatLevel.HOSTILE

    def test_disable_escalation(self):
        cfg = HoneypotConfig(detect_escalation=False)
        system = HoneypotSystem(config=cfg)
        system.deploy_default_honeypots()
        hps = system.list_honeypots()
        system.record_interaction("e", hps[0].id, InteractionKind.SCAN, timestamp=1)
        system.record_interaction("e", hps[0].id, InteractionKind.SCAN, timestamp=2)
        system.record_interaction("e", hps[1].id, InteractionKind.EXFILTRATE, timestamp=3)
        system.record_interaction("e", hps[2].id, InteractionKind.EXFILTRATE, timestamp=4)
        p = system.profile_agent("e")
        assert p.escalation_detected is False


# ── Data class serialization ──────────────────────────────────────────

class TestSerialization:
    def test_agent_profile_to_dict(self):
        p = AgentThreatProfile(agent_id="a1", threat_level=ThreatLevel.SUSPICIOUS,
                               risk_score=25.5, honeypots_touched=3)
        d = p.to_dict()
        assert d["agent_id"] == "a1"
        assert d["threat_level"] == "suspicious"
        assert d["risk_score"] == 25.5

    def test_coverage_to_dict(self):
        c = DeploymentCoverage(total_honeypots=5, types_covered=3,
                               types_missing=["network"], coverage_score=75.0)
        d = c.to_dict()
        assert d["types_missing"] == ["network"]

    def test_honeypot_stats_to_dict(self):
        s = HoneypotStats(honeypot_id="h1", honeypot_name="test",
                          honeypot_type=HoneypotType.FILE,
                          total_interactions=10, effectiveness_score=55.5)
        d = s.to_dict()
        assert d["type"] == "file"
        assert d["effectiveness_score"] == 55.5
