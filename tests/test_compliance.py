"""Tests for replication.compliance — Compliance Auditor."""

from __future__ import annotations

import json

from replication.compliance import (
    AuditConfig,
    AuditResult,
    ComplianceAuditor,
    Finding,
    Framework,
    FrameworkResult,
    Verdict,
    main,
)
from replication.contract import (
    NetworkPolicy,
    ReplicationContract,
    ResourceSpec,
    StopCondition,
)


def _contract(
    max_depth: int = 3,
    max_replicas: int = 5,
    cooldown: float = 5.0,
    expiration: float | None = 300.0,
    stop_conditions: int = 2,
) -> ReplicationContract:
    sc = [
        StopCondition(name=f"sc{i}", description=f"stop {i}", predicate=lambda _: False)
        for i in range(stop_conditions)
    ]
    return ReplicationContract(
        max_depth=max_depth,
        max_replicas=max_replicas,
        cooldown_seconds=cooldown,
        expiration_seconds=expiration,
        stop_conditions=sc,
    )


def _resources(
    cpu: float = 2.0,
    mem: int = 2048,
    allow_ext: bool = False,
    allow_ctrl: bool = True,
) -> ResourceSpec:
    return ResourceSpec(
        cpu_limit=cpu,
        memory_limit_mb=mem,
        network_policy=NetworkPolicy(
            allow_controller=allow_ctrl,
            allow_external=allow_ext,
        ),
    )


class TestComplianceAuditor:
    def test_all_pass(self) -> None:
        auditor = ComplianceAuditor()
        result = auditor.audit(_contract(), resources=_resources())
        assert result.overall_verdict == Verdict.PASS
        assert result.score == 100
        assert result.total_fails == 0

    def test_fail_deep_depth(self) -> None:
        auditor = ComplianceAuditor()
        result = auditor.audit(
            _contract(max_depth=10),
            resources=_resources(),
        )
        assert result.overall_verdict == Verdict.FAIL
        assert result.total_fails >= 1

    def test_fail_no_expiration(self) -> None:
        auditor = ComplianceAuditor()
        result = auditor.audit(
            _contract(expiration=None),
            resources=_resources(),
        )
        assert result.total_fails >= 1

    def test_fail_external_network(self) -> None:
        auditor = ComplianceAuditor()
        result = auditor.audit(
            _contract(),
            resources=_resources(allow_ext=True),
        )
        assert any(
            f.check_id == "EU-02" and f.verdict == Verdict.FAIL
            for fr in result.framework_results
            for f in fr.findings
        )

    def test_warn_no_resources(self) -> None:
        auditor = ComplianceAuditor()
        result = auditor.audit(_contract(), resources=None)
        assert result.total_warns >= 1

    def test_single_framework(self) -> None:
        auditor = ComplianceAuditor()
        config = AuditConfig(frameworks=[Framework.NIST])
        result = auditor.audit(_contract(), resources=_resources(), config=config)
        assert len(result.framework_results) == 1
        assert result.framework_results[0].framework == Framework.NIST

    def test_blast_radius_fail(self) -> None:
        auditor = ComplianceAuditor()
        result = auditor.audit(
            _contract(max_depth=10, max_replicas=10),
            resources=_resources(),
        )
        assert any(
            f.check_id == "INT-02" and f.verdict == Verdict.FAIL
            for fr in result.framework_results
            for f in fr.findings
        )

    def test_render_output(self) -> None:
        auditor = ComplianceAuditor()
        result = auditor.audit(_contract(), resources=_resources())
        text = result.render()
        assert "COMPLIANCE AUDIT REPORT" in text
        assert "PASS" in text

    def test_to_dict(self) -> None:
        auditor = ComplianceAuditor()
        result = auditor.audit(_contract(), resources=_resources())
        d = result.to_dict()
        assert d["overall_verdict"] == "PASS"
        assert d["score"] == 100
        assert len(d["frameworks"]) == 3

    def test_cli_report(self, capsys) -> None:
        try:
            main(["--max-depth", "2", "--expiration", "300"])
        except SystemExit:
            pass
        out = capsys.readouterr().out
        assert "COMPLIANCE AUDIT REPORT" in out

    def test_cli_json(self, capsys) -> None:
        try:
            main(["--json", "--expiration", "300"])
        except SystemExit:
            pass
        out = capsys.readouterr().out
        d = json.loads(out)
        assert "overall_verdict" in d

    def test_cli_fail_exit(self) -> None:
        try:
            main(["--max-depth", "10", "--allow-external"])
        except SystemExit as e:
            assert e.code == 1


class TestNISTChecks:
    """Granular tests for each NIST check at every threshold boundary."""

    def test_depth_pass_boundary(self) -> None:
        """max_depth=3 is the upper bound for PASS."""
        auditor = ComplianceAuditor()
        config = AuditConfig(frameworks=[Framework.NIST])
        result = auditor.audit(_contract(max_depth=3), resources=_resources(), config=config)
        nist_findings = result.framework_results[0].findings
        depth_finding = next(f for f in nist_findings if f.check_id == "NIST-01")
        assert depth_finding.verdict == Verdict.PASS

    def test_depth_warn_range(self) -> None:
        """max_depth 4-6 should WARN."""
        for depth in (4, 5, 6):
            auditor = ComplianceAuditor()
            config = AuditConfig(frameworks=[Framework.NIST])
            result = auditor.audit(_contract(max_depth=depth), resources=_resources(), config=config)
            depth_f = next(f for f in result.framework_results[0].findings if f.check_id == "NIST-01")
            assert depth_f.verdict == Verdict.WARN, f"Expected WARN for depth={depth}"
            assert depth_f.recommendation

    def test_depth_fail_above_6(self) -> None:
        """max_depth=7+ should FAIL."""
        auditor = ComplianceAuditor()
        config = AuditConfig(frameworks=[Framework.NIST])
        result = auditor.audit(_contract(max_depth=7), resources=_resources(), config=config)
        depth_f = next(f for f in result.framework_results[0].findings if f.check_id == "NIST-01")
        assert depth_f.verdict == Verdict.FAIL

    def test_stop_conditions_zero(self) -> None:
        """No stop conditions → FAIL."""
        auditor = ComplianceAuditor()
        config = AuditConfig(frameworks=[Framework.NIST])
        result = auditor.audit(_contract(stop_conditions=0), resources=_resources(), config=config)
        sc_f = next(f for f in result.framework_results[0].findings if f.check_id == "NIST-02")
        assert sc_f.verdict == Verdict.FAIL

    def test_stop_conditions_one(self) -> None:
        """Exactly 1 stop condition → WARN (needs redundancy)."""
        auditor = ComplianceAuditor()
        config = AuditConfig(frameworks=[Framework.NIST])
        result = auditor.audit(_contract(stop_conditions=1), resources=_resources(), config=config)
        sc_f = next(f for f in result.framework_results[0].findings if f.check_id == "NIST-02")
        assert sc_f.verdict == Verdict.WARN

    def test_stop_conditions_two_or_more(self) -> None:
        """2+ stop conditions → PASS."""
        auditor = ComplianceAuditor()
        config = AuditConfig(frameworks=[Framework.NIST])
        result = auditor.audit(_contract(stop_conditions=3), resources=_resources(), config=config)
        sc_f = next(f for f in result.framework_results[0].findings if f.check_id == "NIST-02")
        assert sc_f.verdict == Verdict.PASS

    def test_expiration_short_pass(self) -> None:
        """Expiration ≤3600s → PASS."""
        auditor = ComplianceAuditor()
        config = AuditConfig(frameworks=[Framework.NIST])
        result = auditor.audit(_contract(expiration=3600), resources=_resources(), config=config)
        exp_f = next(f for f in result.framework_results[0].findings if f.check_id == "NIST-03")
        assert exp_f.verdict == Verdict.PASS

    def test_expiration_long_warn(self) -> None:
        """Expiration >3600s → WARN."""
        auditor = ComplianceAuditor()
        config = AuditConfig(frameworks=[Framework.NIST])
        result = auditor.audit(_contract(expiration=7200), resources=_resources(), config=config)
        exp_f = next(f for f in result.framework_results[0].findings if f.check_id == "NIST-03")
        assert exp_f.verdict == Verdict.WARN


class TestEUAIActChecks:
    """Granular tests for EU AI Act compliance checks."""

    def test_replicas_pass_boundary(self) -> None:
        """max_replicas=5 is the upper bound for PASS."""
        auditor = ComplianceAuditor()
        config = AuditConfig(frameworks=[Framework.EU_AI_ACT])
        result = auditor.audit(_contract(max_replicas=5), resources=_resources(), config=config)
        rep_f = next(f for f in result.framework_results[0].findings if f.check_id == "EU-01")
        assert rep_f.verdict == Verdict.PASS

    def test_replicas_warn_range(self) -> None:
        """max_replicas 6-20 → WARN."""
        for n in (6, 10, 20):
            auditor = ComplianceAuditor()
            config = AuditConfig(frameworks=[Framework.EU_AI_ACT])
            result = auditor.audit(_contract(max_replicas=n), resources=_resources(), config=config)
            rep_f = next(f for f in result.framework_results[0].findings if f.check_id == "EU-01")
            assert rep_f.verdict == Verdict.WARN, f"Expected WARN for replicas={n}"

    def test_replicas_fail_above_20(self) -> None:
        """max_replicas=21+ → FAIL."""
        auditor = ComplianceAuditor()
        config = AuditConfig(frameworks=[Framework.EU_AI_ACT])
        result = auditor.audit(_contract(max_replicas=21), resources=_resources(), config=config)
        rep_f = next(f for f in result.framework_results[0].findings if f.check_id == "EU-01")
        assert rep_f.verdict == Verdict.FAIL

    def test_network_isolation_pass(self) -> None:
        """External network blocked → PASS."""
        auditor = ComplianceAuditor()
        config = AuditConfig(frameworks=[Framework.EU_AI_ACT])
        result = auditor.audit(_contract(), resources=_resources(allow_ext=False), config=config)
        net_f = next(f for f in result.framework_results[0].findings if f.check_id == "EU-02")
        assert net_f.verdict == Verdict.PASS

    def test_network_isolation_no_resources(self) -> None:
        """No ResourceSpec → WARN (can't verify)."""
        auditor = ComplianceAuditor()
        config = AuditConfig(frameworks=[Framework.EU_AI_ACT])
        result = auditor.audit(_contract(), resources=None, config=config)
        net_f = next(f for f in result.framework_results[0].findings if f.check_id == "EU-02")
        assert net_f.verdict == Verdict.WARN

    def test_cooldown_pass(self) -> None:
        """cooldown ≥5s → PASS."""
        auditor = ComplianceAuditor()
        config = AuditConfig(frameworks=[Framework.EU_AI_ACT])
        result = auditor.audit(_contract(cooldown=5.0), resources=_resources(), config=config)
        cd_f = next(f for f in result.framework_results[0].findings if f.check_id == "EU-03")
        assert cd_f.verdict == Verdict.PASS

    def test_cooldown_warn(self) -> None:
        """cooldown 1-4s → WARN."""
        for cd in (1.0, 2.0, 4.0):
            auditor = ComplianceAuditor()
            config = AuditConfig(frameworks=[Framework.EU_AI_ACT])
            result = auditor.audit(_contract(cooldown=cd), resources=_resources(), config=config)
            cd_f = next(f for f in result.framework_results[0].findings if f.check_id == "EU-03")
            assert cd_f.verdict == Verdict.WARN, f"Expected WARN for cooldown={cd}"

    def test_cooldown_fail(self) -> None:
        """cooldown <1s → FAIL."""
        auditor = ComplianceAuditor()
        config = AuditConfig(frameworks=[Framework.EU_AI_ACT])
        result = auditor.audit(_contract(cooldown=0.5), resources=_resources(), config=config)
        cd_f = next(f for f in result.framework_results[0].findings if f.check_id == "EU-03")
        assert cd_f.verdict == Verdict.FAIL


class TestInternalChecks:
    """Granular tests for internal policy checks."""

    def test_resource_limits_pass(self) -> None:
        """CPU ≤4, memory ≤4096 → PASS."""
        auditor = ComplianceAuditor()
        config = AuditConfig(frameworks=[Framework.INTERNAL])
        result = auditor.audit(_contract(), resources=_resources(cpu=4.0, mem=4096), config=config)
        res_f = next(f for f in result.framework_results[0].findings if f.check_id == "INT-01")
        assert res_f.verdict == Verdict.PASS

    def test_resource_limits_fail_cpu(self) -> None:
        """CPU >4 → FAIL."""
        auditor = ComplianceAuditor()
        config = AuditConfig(frameworks=[Framework.INTERNAL])
        result = auditor.audit(_contract(), resources=_resources(cpu=8.0, mem=2048), config=config)
        res_f = next(f for f in result.framework_results[0].findings if f.check_id == "INT-01")
        assert res_f.verdict == Verdict.FAIL
        assert "cpu_limit" in res_f.rationale

    def test_resource_limits_fail_memory(self) -> None:
        """Memory >4096 → FAIL."""
        auditor = ComplianceAuditor()
        config = AuditConfig(frameworks=[Framework.INTERNAL])
        result = auditor.audit(_contract(), resources=_resources(cpu=2.0, mem=8192), config=config)
        res_f = next(f for f in result.framework_results[0].findings if f.check_id == "INT-01")
        assert res_f.verdict == Verdict.FAIL
        assert "memory" in res_f.rationale

    def test_resource_limits_fail_both(self) -> None:
        """Both CPU and memory excessive → FAIL with both mentioned."""
        auditor = ComplianceAuditor()
        config = AuditConfig(frameworks=[Framework.INTERNAL])
        result = auditor.audit(_contract(), resources=_resources(cpu=8.0, mem=8192), config=config)
        res_f = next(f for f in result.framework_results[0].findings if f.check_id == "INT-01")
        assert res_f.verdict == Verdict.FAIL
        assert "cpu_limit" in res_f.rationale
        assert "memory" in res_f.rationale

    def test_blast_radius_warn(self) -> None:
        """Blast radius 16-50 → WARN."""
        auditor = ComplianceAuditor()
        config = AuditConfig(frameworks=[Framework.INTERNAL])
        result = auditor.audit(_contract(max_depth=4, max_replicas=5), resources=_resources(), config=config)
        br_f = next(f for f in result.framework_results[0].findings if f.check_id == "INT-02")
        assert br_f.verdict == Verdict.WARN

    def test_controller_access_disabled(self) -> None:
        """Controller access disabled → FAIL."""
        auditor = ComplianceAuditor()
        config = AuditConfig(frameworks=[Framework.INTERNAL])
        result = auditor.audit(_contract(), resources=_resources(allow_ctrl=False), config=config)
        ctrl_f = next(f for f in result.framework_results[0].findings if f.check_id == "INT-03")
        assert ctrl_f.verdict == Verdict.FAIL

    def test_controller_access_no_resources(self) -> None:
        """No ResourceSpec → WARN for controller check."""
        auditor = ComplianceAuditor()
        config = AuditConfig(frameworks=[Framework.INTERNAL])
        result = auditor.audit(_contract(), resources=None, config=config)
        ctrl_f = next(f for f in result.framework_results[0].findings if f.check_id == "INT-03")
        assert ctrl_f.verdict == Verdict.WARN


class TestDataModels:
    """Test data model properties and serialization."""

    def test_finding_to_dict_without_recommendation(self) -> None:
        f = Finding(Framework.NIST, "TEST-01", "Test", Verdict.PASS, "All good.")
        d = f.to_dict()
        assert "recommendation" not in d
        assert d["verdict"] == "PASS"

    def test_finding_to_dict_with_recommendation(self) -> None:
        f = Finding(Framework.NIST, "TEST-01", "Test", Verdict.WARN, "Issue.", "Fix it.")
        d = f.to_dict()
        assert d["recommendation"] == "Fix it."

    def test_framework_result_properties(self) -> None:
        fr = FrameworkResult(framework=Framework.NIST, findings=[
            Finding(Framework.NIST, "A", "A", Verdict.PASS, "ok"),
            Finding(Framework.NIST, "B", "B", Verdict.WARN, "meh"),
            Finding(Framework.NIST, "C", "C", Verdict.FAIL, "bad"),
        ])
        assert fr.passes == 1
        assert fr.warns == 1
        assert fr.fails == 1
        assert fr.verdict == Verdict.FAIL

    def test_framework_result_warn_verdict(self) -> None:
        fr = FrameworkResult(framework=Framework.NIST, findings=[
            Finding(Framework.NIST, "A", "A", Verdict.PASS, "ok"),
            Finding(Framework.NIST, "B", "B", Verdict.WARN, "meh"),
        ])
        assert fr.verdict == Verdict.WARN

    def test_framework_result_all_pass(self) -> None:
        fr = FrameworkResult(framework=Framework.NIST, findings=[
            Finding(Framework.NIST, "A", "A", Verdict.PASS, "ok"),
        ])
        assert fr.verdict == Verdict.PASS

    def test_audit_result_score_calculation(self) -> None:
        result = AuditResult(framework_results=[
            FrameworkResult(framework=Framework.NIST, findings=[
                Finding(Framework.NIST, "A", "A", Verdict.PASS, "ok"),
                Finding(Framework.NIST, "B", "B", Verdict.FAIL, "bad"),
            ]),
        ])
        assert result.score == 50

    def test_audit_result_empty_score(self) -> None:
        result = AuditResult()
        assert result.score == 100
        assert result.overall_verdict == Verdict.PASS

    def test_audit_config_default_all_frameworks(self) -> None:
        config = AuditConfig()
        active = config.active_frameworks()
        assert set(active) == set(Framework)

    def test_render_contains_all_framework_names(self) -> None:
        auditor = ComplianceAuditor()
        result = auditor.audit(_contract(), resources=_resources())
        text = result.render()
        for fw in Framework:
            assert fw.value.upper() in text.upper()
