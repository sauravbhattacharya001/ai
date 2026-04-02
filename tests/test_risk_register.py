"""Tests for risk_register module."""

from __future__ import annotations

import json
import os
import tempfile

from replication.risk_register import (
    AuditEntry,
    Mitigation,
    RegisterConfig,
    RiskCategory,
    RiskEntry,
    RiskRegister,
    RiskStatus,
)


def _make_entry(**kw):
    defaults = dict(
        risk_id="RISK-001",
        title="Test risk",
        description="A test risk",
        category=RiskCategory.REPLICATION,
        likelihood=3,
        impact=4,
    )
    defaults.update(kw)
    return RiskEntry(**defaults)


class TestRiskEntry:
    def test_inherent_score(self):
        e = _make_entry(likelihood=3, impact=4)
        assert e.inherent_score == 12

    def test_residual_score_no_mitigations(self):
        e = _make_entry(likelihood=3, impact=4)
        assert e.residual_score == e.inherent_score

    def test_residual_score_with_mitigation(self):
        e = _make_entry(likelihood=4, impact=5)
        e.mitigations.append(Mitigation(description="test", effectiveness=0.5, status="Implemented"))
        assert e.residual_score < e.inherent_score

    def test_risk_level(self):
        assert _make_entry(likelihood=5, impact=5).risk_level == "Critical"
        assert _make_entry(likelihood=3, impact=4).risk_level == "High"
        assert _make_entry(likelihood=2, impact=3).risk_level == "Medium"
        assert _make_entry(likelihood=1, impact=2).risk_level == "Low"

    def test_transition_valid(self):
        e = _make_entry()
        e.transition(RiskStatus.ASSESSED)
        assert e.status == RiskStatus.ASSESSED
        assert len(e.audit_trail) == 1

    def test_transition_invalid(self):
        e = _make_entry()
        try:
            e.transition(RiskStatus.MITIGATING)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_add_mitigation(self):
        e = _make_entry()
        e.add_mitigation(Mitigation(description="Fix it", effectiveness=0.3))
        assert len(e.mitigations) == 1
        assert len(e.audit_trail) == 1

    def test_serialization_roundtrip(self):
        e = _make_entry()
        e.mitigations.append(Mitigation(description="mit", effectiveness=0.4, status="Implemented"))
        e.audit_trail.append(AuditEntry(timestamp="2026-01-01T00:00:00", action="test", details="test"))
        d = e.to_dict()
        e2 = RiskEntry.from_dict(d)
        assert e2.risk_id == e.risk_id
        assert e2.category == e.category
        assert len(e2.mitigations) == 1


class TestRiskRegister:
    def test_populate(self):
        reg = RiskRegister(RegisterConfig(agent_count=5, seed=42))
        reg.populate_from_simulation()
        assert len(reg.risks) > 0

    def test_statistics(self):
        reg = RiskRegister(RegisterConfig(seed=42))
        reg.populate_from_simulation()
        stats = reg.statistics()
        assert stats["total_risks"] > 0
        assert "by_level" in stats

    def test_top_risks(self):
        reg = RiskRegister(RegisterConfig(seed=42))
        reg.populate_from_simulation()
        top = reg.top_risks(3)
        assert len(top) <= 3
        if len(top) > 1:
            assert top[0].residual_score >= top[1].residual_score

    def test_summary_output(self):
        reg = RiskRegister(RegisterConfig(seed=42))
        reg.populate_from_simulation()
        s = reg.summary()
        assert "RISK REGISTER" in s

    def test_tabular_output(self):
        reg = RiskRegister(RegisterConfig(seed=42))
        reg.populate_from_simulation()
        t = reg.tabular()
        assert "RISK-" in t

    def test_json_export_import(self):
        reg = RiskRegister(RegisterConfig(seed=42))
        reg.populate_from_simulation()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            f.write(reg.to_json())
            path = f.name
        try:
            reg2 = RiskRegister.from_json(path)
            assert len(reg2.risks) == len(reg.risks)
        finally:
            os.unlink(path)

    def test_csv_export(self):
        reg = RiskRegister(RegisterConfig(seed=42))
        reg.populate_from_simulation()
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            reg.export_csv(path)
            with open(path) as f:
                content = f.read()
            assert "Risk ID" in content
        finally:
            os.unlink(path)

    def test_html_output(self):
        reg = RiskRegister(RegisterConfig(seed=42))
        reg.populate_from_simulation()
        html = reg.to_html()
        assert "Risk Register" in html
        assert "RISK-" in html
