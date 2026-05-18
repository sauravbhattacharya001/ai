"""Tests for the SeverityClassifier module."""

from __future__ import annotations

import io
import json
import sys
import pytest

from replication.severity_classifier import (
    CONTROL_BYPASS_SCORES,
    DATA_SENSITIVITY_SCORES,
    DimensionScore,
    IMPACT_SCOPE_SCORES,
    INTENT_SCORES,
    IncidentReport,
    REVERSIBILITY_SCORES,
    Severity,
    SeverityClassifier,
    VELOCITY_SCORES,
    main,
)


# ---------------------------------------------------------------------------
# Severity enum
# ---------------------------------------------------------------------------


class TestSeverityEnum:
    def test_order(self):
        assert Severity.P0 < Severity.P1 < Severity.P2 < Severity.P3 < Severity.P4

    def test_labels(self):
        assert Severity.P0.label == "CRITICAL"
        assert Severity.P4.label == "INFORMATIONAL"

    def test_response_windows_match_each_level(self):
        assert Severity.P0.response_window == "Immediate"
        assert Severity.P1.response_window == "Within 1 hour"
        assert Severity.P2.response_window == "Within 4 hours"
        assert Severity.P3.response_window == "Within 24 hours"
        assert Severity.P4.response_window == "Next triage meeting"


# ---------------------------------------------------------------------------
# Scoring tables sanity
# ---------------------------------------------------------------------------


class TestScoringTables:
    @pytest.mark.parametrize("table", [
        IMPACT_SCOPE_SCORES,
        DATA_SENSITIVITY_SCORES,
        CONTROL_BYPASS_SCORES,
        REVERSIBILITY_SCORES,
        VELOCITY_SCORES,
        INTENT_SCORES,
    ])
    def test_scores_are_nonnegative(self, table):
        assert all(v >= 0 for v in table.values())

    def test_kill_switch_outranks_logging_bypass(self):
        assert CONTROL_BYPASS_SCORES["kill_switch"] > CONTROL_BYPASS_SCORES["logging"]

    def test_credentials_outranks_public_data(self):
        assert DATA_SENSITIVITY_SCORES["credentials"] > DATA_SENSITIVITY_SCORES["public"]


# ---------------------------------------------------------------------------
# Classifier core behavior
# ---------------------------------------------------------------------------


class TestClassifier:
    def setup_method(self):
        self.c = SeverityClassifier()

    def test_default_classification_is_p4(self):
        # Default intent="ambiguous" contributes a small score, but
        # everything else is zero so we stay well within the P4 bucket.
        report = self.c.classify(description="nothing happening")
        assert report.severity == Severity.P4
        assert report.score == INTENT_SCORES["ambiguous"]

    def test_unknown_dimension_values_score_zero(self):
        # If a caller passes an unknown enum-like string, we should not crash;
        # the dimension contributes 0 and total stays at 0.
        report = self.c.classify(
            description="x",
            impact_scope="not_a_real_value",
            data_sensitivity="not_a_real_value",
            control_bypass=["not_a_real_value"],
            reversibility="not_a_real_value",
            velocity="not_a_real_value",
            intent="not_a_real_value",
        )
        assert report.score == 0
        assert report.severity == Severity.P4

    def test_high_severity_full_blowout(self):
        report = self.c.classify(
            description="Agent bypassed kill switch and replicated to 5 nodes",
            impact_scope="external",
            data_sensitivity="credentials",
            control_bypass=list(CONTROL_BYPASS_SCORES.keys()),
            reversibility="none",
            velocity="exponential",
            intent="deliberate",
        )
        assert report.severity == Severity.P0
        assert report.score == report.max_possible
        assert report.percentage == pytest.approx(100.0)
        # Recommendations for P0 must include incident command + fleet quarantine
        joined = " ".join(report.recommended_actions).lower()
        assert "incident command" in joined
        assert "fleet quarantine" in joined

    def test_control_bypass_score_is_additive(self):
        report = self.c.classify(
            description="combo",
            control_bypass=["kill_switch", "logging"],
        )
        bypass_dim = next(d for d in report.dimensions if d.dimension == "Control Bypass")
        expected = CONTROL_BYPASS_SCORES["kill_switch"] + CONTROL_BYPASS_SCORES["logging"]
        assert bypass_dim.score == expected

    def test_kill_switch_bypass_recommends_secondary_mechanism(self):
        report = self.c.classify(
            description="kill switch evaded",
            control_bypass=["kill_switch"],
        )
        assert any(
            "kill-switch bypass vector" in a or "secondary kill mechanism" in a
            for a in report.recommended_actions
        )

    def test_quarantine_bypass_recommends_manual_isolation(self):
        report = self.c.classify(
            description="quarantine evaded",
            control_bypass=["quarantine"],
        )
        assert any(
            "Manually isolate" in a for a in report.recommended_actions
        )

    def test_rapid_velocity_recommends_rate_limit(self):
        report = self.c.classify(
            description="fast spread",
            velocity="rapid",
        )
        assert any(
            "rate limiting" in a.lower() for a in report.recommended_actions
        )

    def test_external_scope_triggers_stakeholder_notice(self):
        report = self.c.classify(
            description="external blast",
            impact_scope="external",
        )
        assert any("downstream" in a.lower() or "stakeholders" in a.lower()
                    for a in report.recommended_actions)

    def test_score_dimensions_match_report(self):
        report = self.c.classify(
            description="test",
            impact_scope="single_system",
            data_sensitivity="pii",
            control_bypass=["rate_limit"],
            reversibility="partial",
            velocity="moderate",
            intent="suspicious",
        )
        # Score is sum of dimension scores
        assert report.score == sum(d.score for d in report.dimensions)
        # All six dimensions should be present
        names = [d.dimension for d in report.dimensions]
        assert names == [
            "Impact Scope",
            "Data Sensitivity",
            "Control Bypass",
            "Reversibility",
            "Velocity",
            "Intent Signal",
        ]

    def test_percentage_in_expected_range(self):
        report = self.c.classify(description="x",
                                   impact_scope="external",
                                   data_sensitivity="pii")
        assert 0.0 <= report.percentage <= 100.0

    @pytest.mark.parametrize("inputs,expected", [
        # P0: very high signal across many dims
        (dict(impact_scope="external",
              data_sensitivity="credentials",
              control_bypass=["kill_switch", "quarantine", "access_control"],
              reversibility="none",
              velocity="exponential",
              intent="deliberate"),
         Severity.P0),
        # P4: nothing
        (dict(), Severity.P4),
    ])
    def test_threshold_buckets(self, inputs, expected):
        report = self.c.classify(description="x", **inputs)
        assert report.severity == expected

    def test_always_includes_timeline_action(self):
        report = self.c.classify(description="anything",
                                   impact_scope="single_agent")
        assert any("incident timeline" in a.lower()
                    for a in report.recommended_actions)


# ---------------------------------------------------------------------------
# Report serialization
# ---------------------------------------------------------------------------


class TestReportSerialization:
    def test_to_dict_roundtrips_via_json(self):
        c = SeverityClassifier()
        report = c.classify(
            description="serialize me",
            impact_scope="multi_agent",
            data_sensitivity="internal",
            control_bypass=["rate_limit"],
            reversibility="partial",
            velocity="slow",
            intent="ambiguous",
        )
        payload = report.to_dict()
        decoded = json.loads(json.dumps(payload))
        assert decoded["severity"] in {s.name for s in Severity}
        assert decoded["score"] == report.score
        assert len(decoded["dimensions"]) == 6
        assert decoded["recommended_actions"] == report.recommended_actions

    def test_summary_contains_key_fields(self):
        c = SeverityClassifier()
        report = c.classify(
            description="hello world",
            impact_scope="single_agent",
            data_sensitivity="internal",
        )
        text = report.summary()
        assert report.severity.name in text
        assert "hello world" in text
        assert "Dimension Breakdown" in text


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestCLI:
    def test_describe_outputs_summary(self, capsys):
        main(["--describe", "agent did a thing", "--impact", "single_agent"])
        out = capsys.readouterr().out
        assert "Incident Severity Report" in out
        assert "agent did a thing" in out

    def test_describe_json_emits_valid_json(self, capsys):
        main([
            "--describe", "boom",
            "--impact", "external",
            "--sensitivity", "credentials",
            "--bypass", "kill_switch", "quarantine",
            "--reversibility", "none",
            "--velocity", "exponential",
            "--intent", "deliberate",
            "--json",
        ])
        out = capsys.readouterr().out
        payload = json.loads(out)
        assert payload["severity"] == "P0"
        assert payload["score"] > 0
        assert any(d["dimension"] == "Control Bypass" for d in payload["dimensions"])

    def test_batch_mode_processes_all_incidents(self, tmp_path, capsys):
        incidents = [
            {"description": "minor", "impact_scope": "single_agent"},
            {"description": "major",
             "impact_scope": "external",
             "data_sensitivity": "credentials",
             "control_bypass": ["kill_switch"],
             "reversibility": "none",
             "velocity": "exponential",
             "intent": "deliberate"},
        ]
        p = tmp_path / "incidents.json"
        p.write_text(json.dumps(incidents))
        main(["--batch", str(p), "--json"])
        out = capsys.readouterr().out
        reports = json.loads(out)
        assert len(reports) == 2
        # The "major" one should be more severe (smaller P-number)
        sev = [Severity[r["severity"]] for r in reports]
        assert sev[1] <= sev[0]

    def test_batch_missing_file_exits_nonzero(self, tmp_path):
        bogus = tmp_path / "nope.json"
        with pytest.raises(SystemExit) as excinfo:
            main(["--batch", str(bogus)])
        assert excinfo.value.code != 0
