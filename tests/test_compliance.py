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
