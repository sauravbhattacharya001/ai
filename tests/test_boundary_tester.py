"""Tests for Capability Boundary Tester."""

from __future__ import annotations

import json
import pytest
from datetime import datetime, timezone

from replication.boundary_tester import (
    BoundaryCategory,
    BoundaryReport,
    BoundarySpec,
    BoundaryTester,
    FaultInjector,
    ProbeResult,
    ProbeVerdict,
    _normalize_path,
    main,
)


# ── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def default_spec():
    return BoundarySpec(
        allowed_paths=["/app", "/tmp"],
        blocked_paths=["/etc/shadow", "/root"],
        allowed_ports=[443, 80],
        blocked_ports=[22, 3306],
        max_memory_mb=512,
        max_processes=10,
        allowed_ipc=["pipe"],
        blocked_ipc=["shared_memory", "socket"],
        allowed_env_vars=["PATH", "HOME"],
        blocked_env_vars=["AWS_SECRET_ACCESS_KEY", "DATABASE_URL"],
    )


@pytest.fixture
def tester():
    return BoundaryTester()


@pytest.fixture
def minimal_spec():
    return BoundarySpec(
        allowed_paths=["/app"],
        blocked_paths=["/root"],
        allowed_ports=[443],
        blocked_ports=[22],
    )


# ── BoundarySpec ─────────────────────────────────────────────────────

class TestBoundarySpec:
    def test_defaults_are_empty_lists(self):
        spec = BoundarySpec()
        assert spec.allowed_paths == []
        assert spec.blocked_paths == []
        assert spec.max_memory_mb == 0.0
        assert spec.max_processes == 0

    def test_custom_values(self, default_spec):
        assert default_spec.allowed_ports == [443, 80]
        assert default_spec.max_memory_mb == 512
        assert default_spec.max_processes == 10


# ── ProbeResult ──────────────────────────────────────────────────────

class TestProbeResult:
    def test_is_leak(self):
        p = ProbeResult(
            probe_id="test", category=BoundaryCategory.FILESYSTEM,
            description="test", target="/etc", expected="block",
            verdict=ProbeVerdict.LEAKED,
        )
        assert p.is_leak
        assert not p.is_over_restricted

    def test_is_over_restricted(self):
        p = ProbeResult(
            probe_id="test", category=BoundaryCategory.NETWORK,
            description="test", target="port 80", expected="allow",
            verdict=ProbeVerdict.DENIED,
        )
        assert p.is_over_restricted
        assert not p.is_leak

    def test_to_dict(self):
        p = ProbeResult(
            probe_id="abc", category=BoundaryCategory.FILESYSTEM,
            description="desc", target="/tmp", expected="allow",
            verdict=ProbeVerdict.ALLOWED, detail="ok",
        )
        d = p.to_dict()
        assert d["probe_id"] == "abc"
        assert d["category"] == "filesystem"
        assert d["verdict"] == "allowed"
        assert "timestamp" in d

    def test_held_is_not_leak(self):
        p = ProbeResult(
            probe_id="x", category=BoundaryCategory.NETWORK,
            description="t", target="22", expected="block",
            verdict=ProbeVerdict.HELD,
        )
        assert not p.is_leak
        assert not p.is_over_restricted


# ── BoundaryReport ───────────────────────────────────────────────────

class TestBoundaryReport:
    def test_empty_report(self):
        r = BoundaryReport(agent_id="a1", spec=BoundarySpec())
        assert r.total_probes == 0
        assert r.containment_score == 100.0
        assert r.risk_level == "low"
        assert r.leaks == []
        assert r.held == []

    def test_score_with_leak(self):
        r = BoundaryReport(agent_id="a1", spec=BoundarySpec(), probes=[
            ProbeResult("p1", BoundaryCategory.FILESYSTEM, "t", "/x", "block", ProbeVerdict.HELD),
            ProbeResult("p2", BoundaryCategory.FILESYSTEM, "t", "/y", "block", ProbeVerdict.LEAKED),
        ])
        assert r.total_probes == 2
        assert r.containment_score == 50.0
        assert len(r.leaks) == 1
        assert r.risk_level in ("high", "critical")

    def test_score_excludes_errors(self):
        r = BoundaryReport(agent_id="a1", spec=BoundarySpec(), probes=[
            ProbeResult("p1", BoundaryCategory.FILESYSTEM, "t", "/x", "block", ProbeVerdict.HELD),
            ProbeResult("p2", BoundaryCategory.FILESYSTEM, "t", "/y", "block", ProbeVerdict.ERROR),
        ])
        assert r.containment_score == 100.0

    def test_multiple_leaks_is_critical(self):
        leaks = [
            ProbeResult(f"p{i}", BoundaryCategory.FILESYSTEM, "t", f"/{i}", "block", ProbeVerdict.LEAKED)
            for i in range(3)
        ]
        r = BoundaryReport(agent_id="a1", spec=BoundarySpec(), probes=leaks)
        assert r.risk_level == "critical"

    def test_over_restricted_moderate_risk(self):
        r = BoundaryReport(agent_id="a1", spec=BoundarySpec(), probes=[
            ProbeResult("p1", BoundaryCategory.NETWORK, "t", "80", "allow", ProbeVerdict.DENIED),
            ProbeResult("p2", BoundaryCategory.NETWORK, "t", "443", "allow", ProbeVerdict.ALLOWED),
        ])
        assert r.risk_level == "moderate"
        assert len(r.over_restricted) == 1

    def test_by_category(self):
        r = BoundaryReport(agent_id="a1", spec=BoundarySpec(), probes=[
            ProbeResult("p1", BoundaryCategory.FILESYSTEM, "t", "/x", "block", ProbeVerdict.HELD),
            ProbeResult("p2", BoundaryCategory.NETWORK, "t", "22", "block", ProbeVerdict.HELD),
            ProbeResult("p3", BoundaryCategory.FILESYSTEM, "t", "/y", "allow", ProbeVerdict.ALLOWED),
        ])
        cats = r.by_category()
        assert len(cats["filesystem"]) == 2
        assert len(cats["network"]) == 1

    def test_summary(self, default_spec):
        tester = BoundaryTester()
        report = tester.test_boundaries("a1", default_spec)
        s = report.summary()
        assert s["agent_id"] == "a1"
        assert "containment_score" in s
        assert "risk_level" in s
        assert "by_category" in s

    def test_render(self, default_spec):
        tester = BoundaryTester()
        report = tester.test_boundaries("a1", default_spec)
        text = report.render()
        assert "Capability Boundary Test Report" in text
        assert "a1" in text
        assert "Score:" in text


# ── BoundaryTester ───────────────────────────────────────────────────

class TestBoundaryTester:
    def test_all_categories_tested(self, tester, default_spec):
        report = tester.test_boundaries("worker-1", default_spec)
        cats = report.by_category()
        assert "filesystem" in cats
        assert "network" in cats
        assert "process" in cats
        assert "memory" in cats
        assert "ipc" in cats
        assert "environment" in cats

    def test_filesystem_probes(self, tester, default_spec):
        report = tester.test_boundaries("w1", default_spec)
        fs_probes = report.by_category()["filesystem"]
        # Should have probes for blocked paths, allowed paths,
        # sensitive paths, and traversal
        assert len(fs_probes) >= 4

        # Blocked paths should be HELD
        blocked_probes = [p for p in fs_probes if p.target in default_spec.blocked_paths]
        for p in blocked_probes:
            assert p.verdict == ProbeVerdict.HELD

        # Allowed paths should be ALLOWED
        allowed_probes = [p for p in fs_probes if p.target in default_spec.allowed_paths]
        for p in allowed_probes:
            assert p.verdict == ProbeVerdict.ALLOWED

    def test_network_probes(self, tester, default_spec):
        report = tester.test_boundaries("w1", default_spec)
        net_probes = report.by_category()["network"]
        assert len(net_probes) >= 4

        # Check blocked ports are HELD
        for port in default_spec.blocked_ports:
            port_probes = [p for p in net_probes if str(port) in p.target]
            assert len(port_probes) >= 1
            assert all(p.verdict == ProbeVerdict.HELD for p in port_probes)

    def test_process_probes(self, tester, default_spec):
        report = tester.test_boundaries("w1", default_spec)
        proc_probes = report.by_category()["process"]
        assert len(proc_probes) == 3  # at limit, over limit, fork bomb

    def test_memory_probes(self, tester, default_spec):
        report = tester.test_boundaries("w1", default_spec)
        mem_probes = report.by_category()["memory"]
        assert len(mem_probes) == 3  # 80%, at limit, 150%

    def test_ipc_probes(self, tester, default_spec):
        report = tester.test_boundaries("w1", default_spec)
        ipc_probes = report.by_category()["ipc"]
        # allowed + blocked + unlisted mechanisms
        assert len(ipc_probes) >= 3

    def test_env_probes(self, tester, default_spec):
        report = tester.test_boundaries("w1", default_spec)
        env_probes = report.by_category()["environment"]
        assert len(env_probes) >= 4

    def test_filter_by_category(self, default_spec):
        tester = BoundaryTester(categories=[BoundaryCategory.NETWORK])
        report = tester.test_boundaries("w1", default_spec)
        cats = report.by_category()
        assert "network" in cats
        assert "filesystem" not in cats
        assert "process" not in cats

    def test_empty_spec(self, tester):
        report = tester.test_boundaries("w1", BoundarySpec())
        assert report.containment_score == 100.0
        assert report.total_probes > 0  # Still tests sensitive paths/ports/env

    def test_perfect_containment_score(self, tester, default_spec):
        report = tester.test_boundaries("w1", default_spec)
        # Default spec should have perfect containment
        assert report.containment_score == 100.0
        assert report.risk_level == "low"
        assert len(report.leaks) == 0

    def test_probe_ids_unique(self, tester, default_spec):
        report = tester.test_boundaries("w1", default_spec)
        ids = [p.probe_id for p in report.probes]
        assert len(ids) == len(set(ids)), "Probe IDs should be unique"

    def test_compare(self, default_spec):
        tester = BoundaryTester()
        spec_tighter = BoundarySpec(
            allowed_paths=["/app"],
            blocked_paths=["/etc/shadow", "/root", "/tmp"],
            allowed_ports=[443],
            blocked_ports=[22, 80, 3306],
            max_memory_mb=256,
            max_processes=5,
            allowed_ipc=["pipe"],
            blocked_ipc=["shared_memory", "socket", "signal"],
            allowed_env_vars=["PATH"],
            blocked_env_vars=["AWS_SECRET_ACCESS_KEY", "DATABASE_URL", "HOME"],
        )
        diff = tester.compare("w1", default_spec, spec_tighter)
        assert "score_before" in diff
        assert "score_after" in diff
        assert "score_delta" in diff
        assert diff["agent_id"] == "w1"


# ── FaultInjector ────────────────────────────────────────────────────

class TestFaultInjector:
    def test_inject_path_leak(self, tester, default_spec):
        injector = FaultInjector(tester)
        report = injector.inject_path_leak("w1", default_spec, "/etc/shadow")
        leaks = report.leaks
        assert len(leaks) >= 1
        leak_targets = [l.target for l in leaks]
        assert "/etc/shadow" in leak_targets

    def test_inject_port_leak(self, tester, default_spec):
        injector = FaultInjector(tester)
        report = injector.inject_port_leak("w1", default_spec, 22)
        leaks = report.leaks
        assert len(leaks) >= 1
        assert any("22" in l.target for l in leaks)

    def test_inject_env_leak(self, tester, default_spec):
        injector = FaultInjector(tester)
        report = injector.inject_env_leak("w1", default_spec, "AWS_SECRET_ACCESS_KEY")
        leaks = report.leaks
        assert len(leaks) >= 1
        assert any(l.target == "AWS_SECRET_ACCESS_KEY" for l in leaks)

    def test_fault_lowers_score(self, tester, default_spec):
        normal = tester.test_boundaries("w1", default_spec)
        injector = FaultInjector(tester)
        faulted = injector.inject_path_leak("w1", default_spec, "/etc/shadow")
        assert faulted.containment_score <= normal.containment_score

    def test_fault_raises_risk(self, tester, default_spec):
        injector = FaultInjector(tester)
        faulted = injector.inject_path_leak("w1", default_spec, "/etc/shadow")
        assert faulted.risk_level in ("high", "critical")


# ── Path Normalization ───────────────────────────────────────────────

class TestNormalizePath:
    def test_simple_path(self):
        assert _normalize_path("/app/data") == "/app/data"

    def test_trailing_slash(self):
        assert _normalize_path("/app/") == "/app"

    def test_dotdot(self):
        assert _normalize_path("/app/../etc") == "/etc"

    def test_deep_traversal(self):
        assert _normalize_path("/app/sub/../../etc/passwd") == "/etc/passwd"

    def test_dot(self):
        assert _normalize_path("/app/./data") == "/app/data"

    def test_backslash(self):
        assert _normalize_path("\\app\\data") == "/app/data"


# ── CLI ──────────────────────────────────────────────────────────────

class TestCLI:
    def test_default_run(self, capsys):
        main(["--agent", "test-agent"])
        out = capsys.readouterr().out
        assert "Capability Boundary Test Report" in out
        assert "test-agent" in out

    def test_json_output(self, capsys):
        main(["--agent", "test-agent", "--json"])
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["agent_id"] == "test-agent"
        assert "containment_score" in data

    def test_category_filter(self, capsys):
        main(["--category", "network", "--json"])
        out = capsys.readouterr().out
        data = json.loads(out)
        assert "network" in data["by_category"]
        cats = set(data["by_category"].keys())
        assert cats == {"network"}

    def test_inject_path_leak_cli(self, capsys):
        main(["--inject-path-leak", "/etc/shadow", "--json"])
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["leaks"] >= 1

    def test_inject_port_leak_cli(self, capsys):
        main(["--inject-port-leak", "22", "--json"])
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["leaks"] >= 1
