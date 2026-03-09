"""Tests for ContainmentVerifier."""
import json, time, pytest
from replication.containment import (
    ContainmentControl, ContainmentPolicy, ContainmentVerifier,
    ControlCategory, ControlSeverity, Finding, FindingStatus, VerificationResult,
)

def _good(aid="agent-001"):
    return {"agent_id": aid,
            "network": {"outbound_allowed": False, "inbound_allowed": False, "dns_allowed": False, "allowed_hosts": [], "open_ports": []},
            "filesystem": {"writable_paths": ["/tmp/a"], "readable_paths": ["/tmp/a","/opt/m"], "denied_paths": ["/etc","/var","/home"], "max_disk_mb": 512, "used_disk_mb": 45},
            "capabilities": {"can_spawn_processes": False, "can_modify_self": False, "can_access_network": False, "can_escalate_privileges": False, "can_access_hardware": False, "allowed_syscalls": ["read","write","open","close","mmap"]},
            "resources": {"max_cpu_percent": 25.0, "max_memory_mb": 1024, "max_open_files": 64, "max_threads": 4, "max_runtime_seconds": 3600},
            "ipc": {"allowed_channels": ["supervisor"], "message_size_limit_kb": 64, "rate_limit_per_second": 10, "encrypted": True}}

def _bad(aid="agent-bad"):
    return {"agent_id": aid,
            "network": {"outbound_allowed": True, "inbound_allowed": True, "dns_allowed": True, "allowed_hosts": ["*"], "open_ports": [8080,443,22]},
            "filesystem": {"writable_paths": ["/","/tmp","/var","/home","/opt"], "readable_paths": ["/"]*15, "denied_paths": [], "max_disk_mb": 10000, "used_disk_mb": 9500},
            "capabilities": {"can_spawn_processes": True, "can_modify_self": True, "can_access_network": True, "can_escalate_privileges": True, "can_access_hardware": True, "allowed_syscalls": []},
            "resources": {"max_cpu_percent": 100.0, "max_memory_mb": 65536, "max_open_files": 1024, "max_threads": 128, "max_runtime_seconds": 999999},
            "ipc": {"allowed_channels": [], "message_size_limit_kb": 99999, "rate_limit_per_second": 99999, "encrypted": False}}

class TestContainmentControl:
    def test_to_dict(self):
        c = ContainmentControl("NET-001","Test",ControlCategory.NETWORK,ControlSeverity.CRITICAL,"d","network.outbound_allowed",comparator="false")
        d = c.to_dict()
        assert d["id"] == "NET-001" and d["category"] == "network"
    def test_default_comparator(self):
        c = ContainmentControl("X","X",ControlCategory.NETWORK,ControlSeverity.LOW,"x","x")
        assert c.comparator == "eq"

class TestFinding:
    def test_to_dict(self):
        ctrl = ContainmentControl("T-1","T",ControlCategory.CAPABILITY,ControlSeverity.HIGH,"d","x")
        f = Finding(control=ctrl, status=FindingStatus.FAIL, actual=True, message="bad")
        assert f.to_dict()["status"] == "fail"

class TestContainmentPolicy:
    def test_strict_categories(self):
        p = ContainmentPolicy.strict()
        assert {c.category for c in p.controls} == {ControlCategory.NETWORK, ControlCategory.FILESYSTEM, ControlCategory.CAPABILITY, ControlCategory.RESOURCE, ControlCategory.IPC}
    def test_strict_count(self):
        assert len(ContainmentPolicy.strict().controls) == 26
    def test_standard_categories(self):
        assert len({c.category for c in ContainmentPolicy.standard().controls}) == 5
    def test_minimal_critical(self):
        p = ContainmentPolicy.minimal()
        for c in p.controls: assert c.severity == ControlSeverity.CRITICAL
        assert len(p.controls) >= 4
    def test_custom(self):
        ctrl = ContainmentControl("C-1","C",ControlCategory.NETWORK,ControlSeverity.LOW,"x","x")
        p = ContainmentPolicy.custom("my", [ctrl])
        assert p.name == "my" and len(p.controls) == 1
    def test_to_dict(self):
        d = ContainmentPolicy.strict().to_dict()
        assert d["name"] == "strict" and len(d["controls"]) == 26

class TestVerificationResult:
    def test_empty(self):
        r = VerificationResult("x", 0)
        assert r.score == 100.0 and r.grade == "A+" and r.total == 0
    def test_all_pass(self):
        c = ContainmentControl("T","T",ControlCategory.NETWORK,ControlSeverity.HIGH,"","x")
        r = VerificationResult("x", 0, findings=[Finding(c, FindingStatus.PASS)])
        assert r.score == 100.0 and r.passed == 1 and r.failed == 0
    def test_all_fail(self):
        c = ContainmentControl("T","T",ControlCategory.NETWORK,ControlSeverity.HIGH,"","x")
        r = VerificationResult("x", 0, findings=[Finding(c, FindingStatus.FAIL, message="bad")])
        assert r.score == 0.0 and r.grade == "F"
    def test_warn_partial(self):
        c = ContainmentControl("T","T",ControlCategory.NETWORK,ControlSeverity.MEDIUM,"","x")
        r = VerificationResult("x", 0, findings=[Finding(c, FindingStatus.WARN, message="meh")])
        assert r.score == 50.0
    def test_skipped(self):
        c = ContainmentControl("T","T",ControlCategory.NETWORK,ControlSeverity.HIGH,"","x")
        r = VerificationResult("x", 0, findings=[Finding(c, FindingStatus.SKIP)])
        assert r.total == 0 and r.skipped == 1 and r.score == 100.0
    def test_critical_failures(self):
        crit = ContainmentControl("C","C",ControlCategory.CAPABILITY,ControlSeverity.CRITICAL,"","x")
        low = ContainmentControl("L","L",ControlCategory.RESOURCE,ControlSeverity.LOW,"","x")
        r = VerificationResult("x", 0, findings=[Finding(crit, FindingStatus.FAIL, message="cf"), Finding(low, FindingStatus.FAIL, message="lf")])
        assert len(r.critical_failures) == 1
    def test_recommendations(self):
        c = ContainmentControl("T","T",ControlCategory.NETWORK,ControlSeverity.HIGH,"","x")
        r = VerificationResult("x", 0, findings=[Finding(c, FindingStatus.FAIL, message="fix"), Finding(c, FindingStatus.PASS)])
        assert len(r.recommendations) == 1
    def test_to_dict(self):
        assert "score" in VerificationResult("a", 123.0, policy_name="strict").to_dict()
    def test_to_json(self):
        assert json.loads(VerificationResult("a", 123.0).to_json())["agent_id"] == "a"
    def test_grade_tiers(self):
        pass

class TestComparators:
    def test_false_pass(self): assert ContainmentVerifier._check("false", False, None)[0] == FindingStatus.PASS
    def test_false_fail(self): assert ContainmentVerifier._check("false", True, None)[0] == FindingStatus.FAIL
    def test_true_pass(self): assert ContainmentVerifier._check("true", True, None)[0] == FindingStatus.PASS
    def test_true_fail(self): assert ContainmentVerifier._check("true", False, None)[0] == FindingStatus.FAIL
    def test_eq_pass(self): assert ContainmentVerifier._check("eq", 42, 42)[0] == FindingStatus.PASS
    def test_eq_fail(self): assert ContainmentVerifier._check("eq", 42, 99)[0] == FindingStatus.FAIL
    def test_ne_pass(self): assert ContainmentVerifier._check("ne", 1, 2)[0] == FindingStatus.PASS
    def test_ne_fail(self): assert ContainmentVerifier._check("ne", 1, 1)[0] == FindingStatus.FAIL
    def test_lt(self):
        assert ContainmentVerifier._check("lt", 5, 10)[0] == FindingStatus.PASS
        assert ContainmentVerifier._check("lt", 10, 5)[0] == FindingStatus.FAIL
    def test_le(self):
        assert ContainmentVerifier._check("le", 10, 10)[0] == FindingStatus.PASS
        assert ContainmentVerifier._check("le", 11, 10)[0] == FindingStatus.FAIL
    def test_gt(self):
        assert ContainmentVerifier._check("gt", 10, 5)[0] == FindingStatus.PASS
        assert ContainmentVerifier._check("gt", 5, 10)[0] == FindingStatus.FAIL
    def test_ge(self):
        assert ContainmentVerifier._check("ge", 10, 10)[0] == FindingStatus.PASS
        assert ContainmentVerifier._check("ge", 9, 10)[0] == FindingStatus.FAIL
    def test_empty_pass(self): assert ContainmentVerifier._check("empty", [], None)[0] == FindingStatus.PASS
    def test_empty_fail(self): assert ContainmentVerifier._check("empty", [1], None)[0] == FindingStatus.FAIL
    def test_empty_non_coll(self): assert ContainmentVerifier._check("empty", 42, None)[0] == FindingStatus.FAIL
    def test_notempty_pass(self): assert ContainmentVerifier._check("notempty", [1], None)[0] == FindingStatus.PASS
    def test_notempty_fail(self): assert ContainmentVerifier._check("notempty", [], None)[0] == FindingStatus.FAIL
    def test_maxlen_pass(self): assert ContainmentVerifier._check("maxlen", [1,2], 5)[0] == FindingStatus.PASS
    def test_maxlen_warn(self): assert ContainmentVerifier._check("maxlen", list(range(6)), 5)[0] == FindingStatus.WARN
    def test_maxlen_skip(self): assert ContainmentVerifier._check("maxlen", 42, 5)[0] == FindingStatus.SKIP
    def test_subset_pass(self): assert ContainmentVerifier._check("subset", ["a","b"], ["a","b","c"])[0] == FindingStatus.PASS
    def test_subset_fail(self): assert ContainmentVerifier._check("subset", ["a","d"], ["a","b","c"])[0] == FindingStatus.FAIL
    def test_unknown(self): assert ContainmentVerifier._check("bogus", 1, 1)[0] == FindingStatus.SKIP
    def test_missing(self):
        from replication.containment import _MISSING
        assert ContainmentVerifier._check("eq", _MISSING, 1)[0] == FindingStatus.SKIP

class TestResolve:
    def test_nested(self): assert ContainmentVerifier._resolve({"a":{"b":{"c":42}}}, "a.b.c") == 42
    def test_missing(self):
        from replication.containment import _MISSING
        assert ContainmentVerifier._resolve({"a":{"b":1}}, "a.c") is _MISSING
    def test_top(self): assert ContainmentVerifier._resolve({"agent_id":"x"}, "agent_id") == "x"

class TestVerifyGood:
    def test_strict_high(self):
        r = ContainmentVerifier(ContainmentPolicy.strict()).verify(_good())
        assert r.score >= 90 and r.grade in ("A+","A","A-") and r.failed == 0
    def test_standard(self):
        r = ContainmentVerifier(ContainmentPolicy.standard()).verify(_good())
        assert r.score >= 90 and r.failed == 0
    def test_minimal(self):
        r = ContainmentVerifier(ContainmentPolicy.minimal()).verify(_good())
        assert r.score == 100.0
    def test_policy_name(self):
        assert ContainmentVerifier(ContainmentPolicy.strict()).verify(_good()).policy_name == "strict"

class TestVerifyBad:
    def test_low_score(self):
        r = ContainmentVerifier(ContainmentPolicy.strict()).verify(_bad())
        assert r.score < 30 and r.failed > 10
    def test_critical_failures(self):
        assert len(ContainmentVerifier(ContainmentPolicy.strict()).verify(_bad()).critical_failures) >= 4
    def test_recommendations(self):
        assert len(ContainmentVerifier(ContainmentPolicy.strict()).verify(_bad()).recommendations) > 5

class TestPartial:
    def test_missing_skips(self):
        r = ContainmentVerifier(ContainmentPolicy.strict()).verify({"agent_id":"p","network":{"outbound_allowed":False}})
        assert r.skipped > 10
    def test_disk_ratio_computed(self):
        s = _good(); s["filesystem"]["used_disk_mb"] = 460; s["filesystem"]["max_disk_mb"] = 512
        r = ContainmentVerifier(ContainmentPolicy.strict()).verify(s)
        f = [f for f in r.findings if f.control.id == "FS-004"][0]
        assert f.status == FindingStatus.PASS
    def test_disk_over_90(self):
        s = _good(); s["filesystem"]["used_disk_mb"] = 500; s["filesystem"]["max_disk_mb"] = 512
        f = [f for f in ContainmentVerifier(ContainmentPolicy.strict()).verify(s).findings if f.control.id == "FS-004"][0]
        assert f.status == FindingStatus.FAIL

class TestCompare:
    def test_same(self):
        d = ContainmentVerifier(ContainmentPolicy.strict()).compare(_good("a"), _good("b"))
        assert d["total_differences"] == 0
    def test_different(self):
        d = ContainmentVerifier(ContainmentPolicy.strict()).compare(_good("a"), _bad("b"))
        assert d["total_differences"] > 5 and d["score_delta"] > 0
    def test_ids(self):
        d = ContainmentVerifier(ContainmentPolicy.strict()).compare(_good("alice"), _bad("bob"))
        assert d["agent_a"]["id"] == "alice" and d["agent_b"]["id"] == "bob"

class TestFleet:
    def test_report(self):
        r = ContainmentVerifier(ContainmentPolicy.strict()).verify_fleet({"g1": _good("g1"), "g2": _good("g2"), "b1": _bad("b1")})
        assert r["fleet_size"] == 3 and r["worst_agent"]["id"] == "b1" and r["total_critical_failures"] > 0
    def test_all_good(self):
        r = ContainmentVerifier(ContainmentPolicy.strict()).verify_fleet({"a": _good("a"), "b": _good("b")})
        assert r["total_critical_failures"] == 0 and r["average_score"] >= 90

class TestReport:
    def test_sections(self):
        v = ContainmentVerifier(ContainmentPolicy.strict())
        t = v.report(v.verify(_good()))
        assert "CONTAINMENT VERIFICATION REPORT" in t and "agent-001" in t
    def test_critical_shown(self):
        v = ContainmentVerifier(ContainmentPolicy.strict())
        assert "CRITICAL FAILURES" in v.report(v.verify(_bad()))
    def test_recommendations(self):
        v = ContainmentVerifier(ContainmentPolicy.strict())
        assert "RECOMMENDATIONS" in v.report(v.verify(_bad()))

class TestHistory:
    def test_records(self):
        v = ContainmentVerifier(ContainmentPolicy.minimal())
        v.verify(_good("a")); v.verify(_bad("b"))
        assert len(v.history) == 2
    def test_trend(self):
        v = ContainmentVerifier(ContainmentPolicy.minimal())
        v.verify(_good("a")); v.verify(_bad("b"))
        t = v.trend()
        assert len(t) == 2 and t[0]["score"] > t[1]["score"]
    def test_export(self):
        v = ContainmentVerifier(ContainmentPolicy.minimal())
        v.verify(_good())
        assert json.loads(v.export_json())[0]["agent_id"] == "agent-001"

class TestEnrich:
    def test_ratio(self):
        assert ContainmentVerifier()._enrich_state({"filesystem":{"max_disk_mb":100,"used_disk_mb":50}})["filesystem"]["disk_usage_ratio"] == 0.5
    def test_no_fs(self):
        e = ContainmentVerifier()._enrich_state({"network":{}})
        assert "disk_usage_ratio" not in e.get("filesystem",{})
    def test_zero_max(self):
        e = ContainmentVerifier()._enrich_state({"filesystem":{"max_disk_mb":0,"used_disk_mb":0}})
        assert "disk_usage_ratio" not in e.get("filesystem",{})

class TestEdge:
    def test_empty_state(self):
        r = ContainmentVerifier(ContainmentPolicy.strict()).verify({})
        assert r.agent_id == "unknown" and r.skipped == 26
    def test_type_error(self):
        assert ContainmentVerifier._check("lt", "abc", 5)[0] == FindingStatus.SKIP
    def test_default_strict(self):
        assert ContainmentVerifier().policy.name == "strict"
    def test_timestamp(self):
        before = time.time()
        r = ContainmentVerifier(ContainmentPolicy.minimal()).verify(_good())
        assert before <= r.timestamp <= time.time()
