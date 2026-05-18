"""Tests for replication.finding_triage."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from replication.finding_triage import (
    FindingTriageAdvisor,
    TriagePlaybookItem,
    TriageReport,
    TriageVerdict,
    _demo_findings,
)
from replication.remediation_planner import Finding


FIXED_NOW = datetime(2026, 5, 18, 17, 0, 0, tzinfo=timezone.utc)


def _fixed_now() -> datetime:
    return FIXED_NOW


def _adv(**kw) -> FindingTriageAdvisor:
    kw.setdefault("now", _fixed_now)
    return FindingTriageAdvisor(**kw)


# ── basics ──


def test_empty_batch_grade_a() -> None:
    rpt = _adv().triage([])
    assert rpt.grade == "A"
    assert rpt.portfolio_band == "CALM"
    assert rpt.summary.startswith("intake quiet")
    # fallback playbook present
    assert any(p.id == "INTAKE_HEALTHY" for p in rpt.playbook)
    assert rpt.counts["HOTFIX_NOW"] == 0


def test_single_critical_yields_hotfix_now() -> None:
    f = Finding(
        name="kill-switch",
        source="scorecard",
        status="fail",
        score=20.0,
        summary="kill switch race during shutdown exposes credentials",
        details={"severity": "critical"},
    )
    rpt = _adv().triage([f])
    v = rpt.verdicts[0]
    assert v.verdict == "HOTFIX_NOW"
    assert v.priority == "P0"
    assert v.sla_hours == 4
    assert "HIGH_SEVERITY" in v.reasons


def test_fp_history_closes_low_severity() -> None:
    f = Finding(
        name="noisy-lint",
        source="custom",
        status="warn",
        summary="cosmetic style issue across config files repeatedly",
        details={"severity": "low"},
    )
    rpt = _adv(
        false_positive_history={"custom": 0.8},
        source_trust={"custom": 0.2},
    ).triage([f])
    v = rpt.verdicts[0]
    assert v.verdict == "CLOSE_AS_FALSE_POSITIVE"
    assert v.priority == "P3"


def test_dedupe_to_existing() -> None:
    existing = Finding(
        name="kill-switch",
        source="scorecard",
        status="fail",
        summary="kill switch race condition during shutdown",
        details={"severity": "critical"},
    )
    new = Finding(
        name="kill-switch-dup",
        source="scorecard",
        status="fail",
        summary="kill switch race condition during shutdown",
        details={"severity": "critical"},
    )
    rpt = _adv(existing_findings=[existing]).triage([new])
    v = rpt.verdicts[0]
    assert v.verdict == "DEDUPE_OF_EXISTING"
    assert v.duplicate_of == "scorecard:kill-switch"
    assert "DEDUPE_MATCH" in v.reasons


def test_dedupe_within_batch() -> None:
    a = Finding(
        name="alpha",
        source="scorecard",
        status="fail",
        summary="auth bypass allows credential leak via cached token",
        details={"severity": "high"},
    )
    b = Finding(
        name="beta",
        source="scorecard",
        status="fail",
        summary="auth bypass allows credential leak via cached token",
        details={"severity": "high"},
    )
    rpt = _adv().triage([a, b])
    assert rpt.verdicts[0].verdict != "DEDUPE_OF_EXISTING"
    assert rpt.verdicts[1].verdict == "DEDUPE_OF_EXISTING"
    assert rpt.verdicts[1].duplicate_of == "scorecard:alpha"


def test_regression_bumps_severity_and_priority() -> None:
    f = Finding(
        name="sla-monitor",
        source="regression",
        status="fail",
        summary="SLA monitor regressed after 4.2.1 rollout windows",
        details={"severity": "high", "regressed": True},
    )
    rpt = _adv().triage([f])
    v = rpt.verdicts[0]
    assert "REGRESSION_DETECTED" in v.reasons
    assert v.reconciled_severity == "critical"
    assert v.verdict == "HOTFIX_NOW"


def test_exploitability_hint_recognized() -> None:
    f = Finding(
        name="rce-path",
        source="quick_scan",
        status="warn",
        summary="possible RCE exploit via crafted prompt injection chain",
        details={"severity": "high"},
    )
    rpt = _adv().triage([f])
    v = rpt.verdicts[0]
    assert "EXPLOITABILITY_HINT" in v.reasons
    assert v.reconciled_severity == "critical"


def test_enrichment_triggered_on_thin_summary() -> None:
    f = Finding(name="x", source="custom", status="warn", summary="oops",
                details={})
    rpt = _adv().triage([f])
    v = rpt.verdicts[0]
    assert v.verdict == "ENRICH_AND_RETRIAGE"
    assert v.priority == "P2"
    assert "CAPTURE_REPRO_STEPS" in v.suggested_enrichment
    assert "ATTACH_LOG_BUNDLE" in v.suggested_enrichment


def test_source_trust_affects_score() -> None:
    f = Finding(
        name="middling",
        source="custom",
        status="warn",
        summary="medium-severity drift in policy thresholds across rules",
        details={"severity": "medium"},
    )
    high_trust = _adv(source_trust={"custom": 1.0}).triage([f]).verdicts[0]
    low_trust = _adv(source_trust={"custom": 0.0}).triage([f]).verdicts[0]
    assert high_trust.triage_score > low_trust.triage_score


def test_risk_appetite_monotonic() -> None:
    findings = [
        Finding(
            name="medium-issue",
            source="quick_scan",
            status="warn",
            summary="medium drift in scoring threshold across runs of size",
            details={"severity": "medium"},
        ),
    ]
    cautious = _adv(risk_appetite="cautious").triage(findings).verdicts[0]
    balanced = _adv(risk_appetite="balanced").triage(findings).verdicts[0]
    aggressive = _adv(risk_appetite="aggressive").triage(findings).verdicts[0]
    assert cautious.triage_score >= balanced.triage_score >= aggressive.triage_score


def test_playbook_p0_first_ordering() -> None:
    findings = _demo_findings()
    rpt = _adv(
        false_positive_history={"custom": 0.5},
        source_trust={"custom": 0.3},
    ).triage(findings)
    priorities = [p.priority for p in rpt.playbook]
    # P0/P1/P2/P3 sequence (allow ties)
    order_map = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
    encoded = [order_map[p] for p in priorities]
    assert encoded == sorted(encoded)


def test_json_byte_stable_with_fixed_now() -> None:
    findings = _demo_findings()
    adv1 = _adv(
        false_positive_history={"custom": 0.5},
        source_trust={"custom": 0.3},
    )
    adv2 = _adv(
        false_positive_history={"custom": 0.5},
        source_trust={"custom": 0.3},
    )
    j1 = adv1.triage(findings).to_json()
    j2 = adv2.triage(findings).to_json()
    assert j1 == j2
    parsed = json.loads(j1)
    assert "verdicts" in parsed
    assert "playbook" in parsed
    assert "counts" in parsed


def test_markdown_sections_present() -> None:
    findings = _demo_findings()
    md = _adv().triage(findings).to_markdown()
    assert "# Finding Triage Report" in md
    assert "## Summary" in md
    assert "## Findings" in md
    assert "## Playbook" in md
    assert "## Insights" in md


def test_text_headline_present() -> None:
    rpt = _adv().triage(_demo_findings())
    text = rpt.to_text()
    assert "FINDING TRIAGE REPORT" in text
    assert "Headline:" in text


def test_owner_hint_mapping() -> None:
    cases = {
        "scorecard": "security_eng",
        "policy_linter": "appsec",
        "regression": "sre",
        "ux": "product",
        "totally-unknown-source": "unknown",
    }
    for src, owner in cases.items():
        f = Finding(
            name="probe",
            source=src,
            status="warn",
            summary="generic finding text that is sufficiently long here",
            details={"severity": "medium"},
        )
        v = _adv().triage([f]).verdicts[0]
        assert v.owner_hint == owner, (src, v.owner_hint)


def test_invalid_risk_appetite_rejected() -> None:
    with pytest.raises(ValueError):
        FindingTriageAdvisor(risk_appetite="reckless")


def test_invalid_dedupe_threshold_rejected() -> None:
    with pytest.raises(ValueError):
        FindingTriageAdvisor(dedupe_jaccard_threshold=0.0)
    with pytest.raises(ValueError):
        FindingTriageAdvisor(dedupe_jaccard_threshold=1.5)


def test_cli_demo_runs() -> None:
    import subprocess, sys
    res = subprocess.run(
        [sys.executable, "-m", "replication", "triage", "--demo", "--format", "json"],
        capture_output=True, text=True, encoding="utf-8",
        env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"},
    )
    assert res.returncode == 0, res.stderr
    parsed = json.loads(res.stdout)
    assert "verdicts" in parsed and "playbook" in parsed


def test_counts_match_verdicts() -> None:
    rpt = _adv().triage(_demo_findings())
    total_from_counts = sum(rpt.counts.values())
    assert total_from_counts == len(rpt.verdicts)
