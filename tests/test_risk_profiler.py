"""Tests for the Agent Risk Profiler module."""

from __future__ import annotations

import json
import sys

import pytest

from replication.risk_profiler import (
    AgentDossier,
    CategoryScore,
    Finding,
    FindingSeverity,
    FindingSource,
    FleetRiskReport,
    Mitigation,
    ProfilerConfig,
    RiskCategory,
    RiskProfiler,
    RiskTier,
    SEVERITY_WEIGHT,
    TIER_THRESHOLDS,
)


# ---------------------------------------------------------------------------
# Unit tests - Finding
# ---------------------------------------------------------------------------


class TestFinding:
    def test_weight(self):
        f = Finding(
            source=FindingSource.KILLCHAIN,
            severity=FindingSeverity.CRITICAL,
            category=RiskCategory.ESCALATION,
            description="test",
        )
        assert f.weight == 25

    def test_weight_low(self):
        f = Finding(
            source=FindingSource.BEHAVIOR,
            severity=FindingSeverity.LOW,
            category=RiskCategory.DECEPTION,
            description="test",
        )
        assert f.weight == 3

    def test_to_dict(self):
        f = Finding(
            source=FindingSource.CANARY,
            severity=FindingSeverity.HIGH,
            category=RiskCategory.EXFILTRATION,
            description="Data exfil detected",
            evidence="Token CAN-001",
            recommendation="Block egress",
        )
        d = f.to_dict()
        assert d["source"] == "canary"
        assert d["severity"] == "high"
        assert d["category"] == "exfiltration"
        assert d["description"] == "Data exfil detected"
        assert d["evidence"] == "Token CAN-001"
        assert d["recommendation"] == "Block egress"


# ---------------------------------------------------------------------------
# Unit tests - AgentDossier
# ---------------------------------------------------------------------------


class TestAgentDossier:
    def test_empty_dossier(self):
        d = AgentDossier(agent_id="agent-0")
        d._compute()
        assert d.overall_score == 0.0
        assert d.tier == RiskTier.MINIMAL
        assert len(d.category_scores) == len(RiskCategory)
        assert all(cs.score == 0.0 for cs in d.category_scores)
        assert d.mitigations == []

    def test_single_critical_finding(self):
        d = AgentDossier(agent_id="agent-0")
        d.findings.append(Finding(
            source=FindingSource.KILLCHAIN,
            severity=FindingSeverity.CRITICAL,
            category=RiskCategory.ESCALATION,
            description="Kill chain 100% complete",
        ))
        d._compute()
        assert d.overall_score > 0
        assert d.tier in (RiskTier.CRITICAL, RiskTier.HIGH, RiskTier.MEDIUM)
        esc = next(cs for cs in d.category_scores if cs.category == RiskCategory.ESCALATION)
        assert esc.score > 0
        assert esc.finding_count == 1

    def test_multiple_findings_different_categories(self):
        d = AgentDossier(agent_id="agent-1")
        d.findings.extend([
            Finding(FindingSource.KILLCHAIN, FindingSeverity.HIGH, RiskCategory.ESCALATION, "Escalation"),
            Finding(FindingSource.CANARY, FindingSeverity.CRITICAL, RiskCategory.EXFILTRATION, "Exfiltration"),
            Finding(FindingSource.TRUST, FindingSeverity.MEDIUM, RiskCategory.COLLUSION, "Collusion"),
        ])
        d._compute()
        assert d.overall_score > 0
        assert len(d.mitigations) > 0
        nonzero = [cs for cs in d.category_scores if cs.score > 0]
        assert len(nonzero) == 3

    def test_tier_assignment_critical(self):
        d = AgentDossier(agent_id="agent-x")
        # Add many critical findings to push score high
        for _ in range(10):
            d.findings.append(Finding(
                FindingSource.KILLCHAIN, FindingSeverity.CRITICAL,
                RiskCategory.ESCALATION, "Critical chain",
            ))
        d._compute()
        assert d.tier in (RiskTier.CRITICAL, RiskTier.HIGH)

    def test_tier_assignment_minimal(self):
        d = AgentDossier(agent_id="agent-y")
        d.findings.append(Finding(
            FindingSource.BEHAVIOR, FindingSeverity.INFO,
            RiskCategory.DECEPTION, "Minor anomaly",
        ))
        d._compute()
        assert d.tier in (RiskTier.MINIMAL, RiskTier.LOW)

    def test_mitigations_generated(self):
        d = AgentDossier(agent_id="agent-m")
        d.findings.extend([
            Finding(FindingSource.CANARY, FindingSeverity.CRITICAL, RiskCategory.EXFILTRATION, "Exfil"),
            Finding(FindingSource.ESCALATION, FindingSeverity.HIGH, RiskCategory.ESCALATION, "Escalation"),
        ])
        d._compute()
        assert len(d.mitigations) > 0
        # Sorted by impact (descending)
        impacts = [m.impact for m in d.mitigations]
        assert impacts == sorted(impacts, reverse=True)

    def test_mitigations_capped_at_10(self):
        d = AgentDossier(agent_id="agent-cap")
        for cat in RiskCategory:
            for _ in range(5):
                d.findings.append(Finding(
                    FindingSource.SIMULATION, FindingSeverity.HIGH,
                    cat, f"Finding in {cat.value}",
                ))
        d._compute()
        assert len(d.mitigations) <= 10

    def test_render_not_empty(self):
        d = AgentDossier(agent_id="agent-r")
        d.findings.append(Finding(
            FindingSource.TRUST, FindingSeverity.HIGH,
            RiskCategory.COLLUSION, "Test finding",
        ))
        d._compute()
        text = d.render()
        assert "agent-r" in text
        assert "RISK BREAKDOWN" in text
        assert "KEY FINDINGS" in text

    def test_to_dict_structure(self):
        d = AgentDossier(agent_id="agent-d")
        d.findings.append(Finding(
            FindingSource.ESCALATION, FindingSeverity.MEDIUM,
            RiskCategory.ESCALATION, "Test",
        ))
        d._compute()
        data = d.to_dict()
        assert data["agent_id"] == "agent-d"
        assert "overall_score" in data
        assert "tier" in data
        assert "findings" in data
        assert "mitigations" in data
        assert "category_scores" in data

    def test_score_capped_at_100(self):
        d = AgentDossier(agent_id="agent-cap")
        for _ in range(50):
            d.findings.append(Finding(
                FindingSource.SIMULATION, FindingSeverity.CRITICAL,
                RiskCategory.ESCALATION, "Overload",
            ))
        d._compute()
        assert d.overall_score <= 100.0


# ---------------------------------------------------------------------------
# Unit tests - FleetRiskReport
# ---------------------------------------------------------------------------


class TestFleetRiskReport:
    def _make_fleet(self, n: int = 4) -> FleetRiskReport:
        dossiers = []
        sevs = [FindingSeverity.CRITICAL, FindingSeverity.HIGH,
                FindingSeverity.MEDIUM, FindingSeverity.LOW]
        for i in range(n):
            d = AgentDossier(agent_id=f"agent-{i}")
            for j in range(i + 1):
                d.findings.append(Finding(
                    FindingSource.SIMULATION, sevs[i % len(sevs)],
                    list(RiskCategory)[i % len(RiskCategory)],
                    f"Finding {j}",
                ))
            d._compute()
            dossiers.append(d)
        report = FleetRiskReport(dossiers=dossiers)
        report._compute()
        return report

    def test_fleet_risk_score(self):
        report = self._make_fleet()
        assert report.fleet_risk_score >= 0

    def test_tier_distribution(self):
        report = self._make_fleet()
        total = sum(report.tier_distribution.values())
        assert total == 4

    def test_peer_percentiles(self):
        report = self._make_fleet()
        pcts = [d.peer_percentile for d in report.dossiers]
        assert min(pcts) >= 0
        assert max(pcts) <= 100

    def test_hotspots(self):
        report = self._make_fleet()
        assert len(report.category_hotspots) == len(RiskCategory)

    def test_total_findings(self):
        report = self._make_fleet()
        assert report.total_findings == sum(len(d.findings) for d in report.dossiers)

    def test_fleet_mitigations(self):
        report = self._make_fleet()
        assert isinstance(report.top_mitigations, list)
        assert len(report.top_mitigations) <= 10

    def test_render_not_empty(self):
        report = self._make_fleet()
        text = report.render()
        assert "FLEET RISK PROFILE" in text
        assert "AGENT RISK DISTRIBUTION" in text
        assert "INDIVIDUAL AGENT DOSSIERS" in text

    def test_to_dict_structure(self):
        report = self._make_fleet()
        data = report.to_dict()
        assert "fleet_risk_score" in data
        assert "fleet_risk_tier" in data
        assert "dossiers" in data
        assert len(data["dossiers"]) == 4

    def test_empty_fleet(self):
        report = FleetRiskReport(dossiers=[])
        report._compute()
        assert report.fleet_risk_score == 0.0
        assert report.total_findings == 0


# ---------------------------------------------------------------------------
# Unit tests - CategoryScore
# ---------------------------------------------------------------------------


class TestCategoryScore:
    def test_to_dict(self):
        cs = CategoryScore(
            category=RiskCategory.REPLICATION,
            score=42.5,
            finding_count=3,
            top_finding="test",
        )
        d = cs.to_dict()
        assert d["category"] == "replication"
        assert d["score"] == 42.5
        assert d["finding_count"] == 3


# ---------------------------------------------------------------------------
# Unit tests - Mitigation
# ---------------------------------------------------------------------------


class TestMitigation:
    def test_to_dict(self):
        m = Mitigation(
            action="Restrict network",
            impact=35.0,
            category=RiskCategory.EXFILTRATION,
            effort="low",
            source_findings=5,
        )
        d = m.to_dict()
        assert d["action"] == "Restrict network"
        assert d["impact"] == 35.0
        assert d["effort"] == "low"


# ---------------------------------------------------------------------------
# Integration tests - RiskProfiler
# ---------------------------------------------------------------------------


class TestRiskProfiler:
    def test_default_analyze(self):
        """Full integration: default config produces valid report."""
        profiler = RiskProfiler(ProfilerConfig(agent_count=4, seed=42))
        report = profiler.analyze()
        assert len(report.dossiers) == 4
        assert report.fleet_risk_score >= 0
        assert report.fleet_risk_tier in RiskTier

    def test_seeded_reproducibility(self):
        """Same seed produces identical results."""
        config = ProfilerConfig(agent_count=4, seed=123)
        r1 = RiskProfiler(config).analyze()
        r2 = RiskProfiler(config).analyze()
        assert r1.fleet_risk_score == r2.fleet_risk_score
        assert r1.total_findings == r2.total_findings
        for d1, d2 in zip(r1.dossiers, r2.dossiers):
            assert d1.overall_score == d2.overall_score
            assert d1.tier == d2.tier

    def test_different_seeds_differ(self):
        """Different seeds produce different results."""
        r1 = RiskProfiler(ProfilerConfig(agent_count=6, seed=1)).analyze()
        r2 = RiskProfiler(ProfilerConfig(agent_count=6, seed=999)).analyze()
        # With enough agents, scores should differ
        scores1 = [d.overall_score for d in r1.dossiers]
        scores2 = [d.overall_score for d in r2.dossiers]
        assert scores1 != scores2

    def test_single_agent(self):
        """Works with just 1 agent."""
        profiler = RiskProfiler(ProfilerConfig(agent_count=1, seed=42))
        report = profiler.analyze()
        assert len(report.dossiers) == 1

    def test_large_fleet(self):
        """Handles larger fleet sizes."""
        profiler = RiskProfiler(ProfilerConfig(agent_count=20, seed=42))
        report = profiler.analyze()
        assert len(report.dossiers) == 20
        total_dist = sum(report.tier_distribution.values())
        assert total_dist == 20

    def test_disabled_modules(self):
        """Can disable individual modules."""
        config = ProfilerConfig(
            agent_count=4, seed=42,
            enable_killchain=False,
            enable_escalation=False,
            enable_behavior=False,
            enable_trust=False,
            enable_canary=False,
            enable_goal_inference=False,
        )
        report = RiskProfiler(config).analyze()
        # All modules disabled = no findings
        assert report.total_findings == 0
        for d in report.dossiers:
            assert d.tier == RiskTier.MINIMAL

    def test_partial_modules(self):
        """Running with only some modules."""
        config = ProfilerConfig(
            agent_count=4, seed=42,
            enable_killchain=True,
            enable_escalation=False,
            enable_behavior=False,
            enable_trust=False,
            enable_canary=False,
            enable_goal_inference=False,
        )
        report = RiskProfiler(config).analyze()
        # Only killchain findings
        for d in report.dossiers:
            for f in d.findings:
                assert f.source == FindingSource.KILLCHAIN

    def test_json_roundtrip(self):
        """to_dict produces valid JSON."""
        report = RiskProfiler(ProfilerConfig(agent_count=3, seed=42)).analyze()
        data = report.to_dict()
        # Should be JSON-serializable
        s = json.dumps(data)
        parsed = json.loads(s)
        assert parsed["agent_count"] == 3

    def test_render_produces_output(self):
        """render() produces non-empty string."""
        report = RiskProfiler(ProfilerConfig(agent_count=3, seed=42)).analyze()
        text = report.render()
        assert len(text) > 100
        assert "FLEET RISK PROFILE" in text

    def test_agent_ids_correct(self):
        """Agent IDs match expected format."""
        report = RiskProfiler(ProfilerConfig(agent_count=5, seed=42)).analyze()
        ids = {d.agent_id for d in report.dossiers}
        expected = {f"agent-{i}" for i in range(5)}
        assert ids == expected


# ---------------------------------------------------------------------------
# Enum / constant tests
# ---------------------------------------------------------------------------


class TestEnums:
    def test_risk_tier_values(self):
        assert RiskTier.CRITICAL.value == "critical"
        assert RiskTier.MINIMAL.value == "minimal"

    def test_risk_category_values(self):
        assert len(RiskCategory) == 7  # all 7 categories

    def test_severity_weights(self):
        assert SEVERITY_WEIGHT[FindingSeverity.CRITICAL] > SEVERITY_WEIGHT[FindingSeverity.HIGH]
        assert SEVERITY_WEIGHT[FindingSeverity.HIGH] > SEVERITY_WEIGHT[FindingSeverity.MEDIUM]
        assert SEVERITY_WEIGHT[FindingSeverity.MEDIUM] > SEVERITY_WEIGHT[FindingSeverity.LOW]
        assert SEVERITY_WEIGHT[FindingSeverity.LOW] > SEVERITY_WEIGHT[FindingSeverity.INFO]

    def test_tier_thresholds_descending(self):
        thresholds = [t for t, _ in TIER_THRESHOLDS]
        assert thresholds == sorted(thresholds, reverse=True)

    def test_finding_sources(self):
        assert len(FindingSource) >= 8  # All module sources


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_zero_agent_count(self):
        config = ProfilerConfig(agent_count=0, seed=42)
        report = RiskProfiler(config).analyze()
        assert len(report.dossiers) == 0
        assert report.fleet_risk_score == 0.0

    def test_dossier_without_compute(self):
        """Dossier defaults work before _compute."""
        d = AgentDossier(agent_id="test")
        assert d.overall_score == 0.0
        assert d.tier == RiskTier.MINIMAL
        assert d.peer_percentile == 0.0

    def test_fleet_single_agent_percentile(self):
        """Single agent gets 0th percentile."""
        d = AgentDossier(agent_id="solo")
        d.findings.append(Finding(
            FindingSource.SIMULATION, FindingSeverity.HIGH,
            RiskCategory.ESCALATION, "test",
        ))
        d._compute()
        report = FleetRiskReport(dossiers=[d])
        report._compute()
        assert d.peer_percentile == 0.0

    def test_all_same_score_percentiles(self):
        """All agents with same score get same percentile."""
        dossiers = []
        for i in range(5):
            d = AgentDossier(agent_id=f"agent-{i}")
            d.findings.append(Finding(
                FindingSource.SIMULATION, FindingSeverity.MEDIUM,
                RiskCategory.DECEPTION, "Same finding",
            ))
            d._compute()
            dossiers.append(d)
        report = FleetRiskReport(dossiers=dossiers)
        report._compute()
        pcts = {d.peer_percentile for d in report.dossiers}
        assert len(pcts) == 1  # All same
