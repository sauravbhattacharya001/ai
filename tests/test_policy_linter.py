"""Tests for the Safety Policy Linter."""

from __future__ import annotations

import json
import unittest
from io import StringIO

from replication.policy import (
    Operator,
    PolicyRule,
    SafetyPolicy,
    Severity,
    POLICY_PRESETS,
)
from replication.policy_linter import (
    FindingCategory,
    FindingSeverity,
    LintFinding,
    LintReport,
    PolicyLinter,
)

class TestLintFinding(unittest.TestCase):
    """Tests for the LintFinding data class."""

    def test_render_basic(self):
        f = LintFinding(
            severity=FindingSeverity.ERROR,
            category=FindingCategory.INVALID_METRIC,
            message="Unknown metric 'foo'",
        )
        text = f.render()
        self.assertIn("🔴", text)
        self.assertIn("invalid_metric", text)
        self.assertIn("Unknown metric 'foo'", text)

    def test_render_with_suggestion(self):
        f = LintFinding(
            severity=FindingSeverity.WARNING,
            category=FindingCategory.THRESHOLD,
            message="Out of range",
            suggestion="Use a value in [0, 1]",
        )
        text = f.render()
        self.assertIn("↳", text)
        self.assertIn("Use a value in [0, 1]", text)

    def test_to_dict(self):
        f = LintFinding(
            severity=FindingSeverity.INFO,
            category=FindingCategory.BEST_PRACTICE,
            message="Hint",
            rule_index=2,
            rule_expr="denial_rate >= 0.3",
            suggestion="Add it",
        )
        d = f.to_dict()
        self.assertEqual(d["severity"], "info")
        self.assertEqual(d["category"], "best_practice")
        self.assertEqual(d["rule_index"], 2)
        self.assertEqual(d["suggestion"], "Add it")

class TestLintReport(unittest.TestCase):
    """Tests for the LintReport data class."""

    def test_empty_report(self):
        r = LintReport(policy_name="Test", rules_checked=3)
        self.assertTrue(r.passed)
        self.assertEqual(r.verdict, "CLEAN")
        self.assertIn("✨", r.render())

    def test_error_fails(self):
        r = LintReport(policy_name="Test", rules_checked=1, findings=[
            LintFinding(FindingSeverity.ERROR, FindingCategory.INVALID_METRIC, "bad"),
        ])
        self.assertFalse(r.passed)
        self.assertEqual(r.verdict, "FAIL")

    def test_warning_only(self):
        r = LintReport(policy_name="Test", rules_checked=1, findings=[
            LintFinding(FindingSeverity.WARNING, FindingCategory.THRESHOLD, "meh"),
        ])
        self.assertTrue(r.passed)
        self.assertEqual(r.verdict, "WARN")

    def test_info_only(self):
        r = LintReport(policy_name="Test", rules_checked=1, findings=[
            LintFinding(FindingSeverity.INFO, FindingCategory.BEST_PRACTICE, "hint"),
        ])
        self.assertTrue(r.passed)
        self.assertEqual(r.verdict, "OK")

    def test_to_dict(self):
        r = LintReport(policy_name="X", rules_checked=5, findings=[
            LintFinding(FindingSeverity.ERROR, FindingCategory.INVALID_METRIC, "a"),
            LintFinding(FindingSeverity.WARNING, FindingCategory.THRESHOLD, "b"),
        ])
        d = r.to_dict()
        self.assertEqual(d["policy_name"], "X")
        self.assertFalse(d["passed"])
        self.assertEqual(d["counts"]["errors"], 1)
        self.assertEqual(d["counts"]["warnings"], 1)
        self.assertEqual(len(d["findings"]), 2)

    def test_summary(self):
        r = LintReport(policy_name="X", rules_checked=2, findings=[
            LintFinding(FindingSeverity.ERROR, FindingCategory.INVALID_METRIC, "a"),
            LintFinding(FindingSeverity.STYLE, FindingCategory.REDUNDANCY, "b"),
        ])
        s = r.summary()
        self.assertIn("1 errors", s)
        self.assertIn("1 style", s)

class TestPolicyLinter(unittest.TestCase):
    """Tests for the PolicyLinter analysis checks."""

    def setUp(self):
        self.linter = PolicyLinter()

    def test_empty_policy(self):
        policy = SafetyPolicy("Empty", [])
        report = self.linter.lint(policy)
        self.assertEqual(report.rules_checked, 0)
        self.assertTrue(any(f.category == FindingCategory.COVERAGE_GAP for f in report.findings))

    def test_invalid_metric(self):
        policy = SafetyPolicy("Bad", [
            PolicyRule("nonexistent_metric", Operator.LE, 10),
        ])
        report = self.linter.lint(policy)
        errors = [f for f in report.findings if f.category == FindingCategory.INVALID_METRIC]
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0].severity, FindingSeverity.ERROR)

    def test_invalid_metric_close_match(self):
        policy = SafetyPolicy("Typo", [
            PolicyRule("total_worker", Operator.LE, 10),  # missing 's'
        ])
        report = self.linter.lint(policy)
        errors = [f for f in report.findings if f.category == FindingCategory.INVALID_METRIC]
        self.assertTrue(len(errors) >= 1)
        self.assertIn("total_workers", errors[0].suggestion)

    def test_contradiction(self):
        policy = SafetyPolicy("Contradiction", [
            PolicyRule("total_workers", Operator.LE, 5),
            PolicyRule("total_workers", Operator.GE, 10),
        ])
        report = self.linter.lint(policy)
        contras = [f for f in report.findings if f.category == FindingCategory.CONTRADICTION]
        self.assertTrue(len(contras) >= 1)

    def test_no_contradiction_when_compatible(self):
        policy = SafetyPolicy("OK", [
            PolicyRule("total_workers", Operator.LE, 50),
            PolicyRule("total_workers", Operator.GE, 5),
        ])
        report = self.linter.lint(policy)
        contras = [f for f in report.findings if f.category == FindingCategory.CONTRADICTION]
        self.assertEqual(len(contras), 0)

    def test_ratio_out_of_range(self):
        policy = SafetyPolicy("BadRange", [
            PolicyRule("denial_rate", Operator.GE, 1.5),
            PolicyRule("total_workers", Operator.LE, 50),
            PolicyRule("max_depth_reached", Operator.LE, 5),
        ])
        report = self.linter.lint(policy)
        threshold = [f for f in report.findings if f.category == FindingCategory.THRESHOLD
                     and "outside [0, 1]" in f.message]
        self.assertTrue(len(threshold) >= 1)

    def test_negative_count(self):
        policy = SafetyPolicy("Neg", [
            PolicyRule("total_workers", Operator.LE, -5),
            PolicyRule("max_depth_reached", Operator.LE, 5),
        ])
        report = self.linter.lint(policy)
        negs = [f for f in report.findings if "Negative" in f.message]
        self.assertTrue(len(negs) >= 1)

    def test_very_permissive_workers(self):
        policy = SafetyPolicy("Permissive", [
            PolicyRule("total_workers", Operator.LE, 1000),
            PolicyRule("max_depth_reached", Operator.LE, 5),
        ])
        report = self.linter.lint(policy)
        perms = [f for f in report.findings if "permissive worker" in f.message.lower()]
        self.assertTrue(len(perms) >= 1)

    def test_very_permissive_depth(self):
        policy = SafetyPolicy("DeepPermissive", [
            PolicyRule("total_workers", Operator.LE, 50),
            PolicyRule("max_depth_reached", Operator.LE, 25),
        ])
        report = self.linter.lint(policy)
        perms = [f for f in report.findings if "permissive depth" in f.message.lower()]
        self.assertTrue(len(perms) >= 1)

    def test_wrong_direction_operator(self):
        policy = SafetyPolicy("WrongDir", [
            PolicyRule("total_workers", Operator.GE, 50),
            PolicyRule("max_depth_reached", Operator.LE, 5),
        ])
        report = self.linter.lint(policy)
        dirs = [f for f in report.findings if "upper bound" in f.message]
        self.assertTrue(len(dirs) >= 1)

    def test_coverage_gap_no_workers(self):
        policy = SafetyPolicy("NoWorkers", [
            PolicyRule("max_depth_reached", Operator.LE, 5),
            PolicyRule("denial_rate", Operator.GE, 0.3),
        ])
        report = self.linter.lint(policy)
        gaps = [f for f in report.findings if f.category == FindingCategory.COVERAGE_GAP
                and "total_workers" in f.message]
        self.assertTrue(len(gaps) >= 1)

    def test_coverage_gap_no_depth(self):
        policy = SafetyPolicy("NoDepth", [
            PolicyRule("total_workers", Operator.LE, 50),
            PolicyRule("denial_rate", Operator.GE, 0.3),
        ])
        report = self.linter.lint(policy)
        gaps = [f for f in report.findings if f.category == FindingCategory.COVERAGE_GAP
                and "max_depth_reached" in f.message]
        self.assertTrue(len(gaps) >= 1)

    def test_no_mc_suggestion(self):
        policy = SafetyPolicy("NoMC", [
            PolicyRule("total_workers", Operator.LE, 50),
            PolicyRule("max_depth_reached", Operator.LE, 5),
            PolicyRule("denial_rate", Operator.GE, 0.3),
            PolicyRule("depth_utilization", Operator.LE, 0.8),
        ])
        report = self.linter.lint(policy)
        mc_hints = [f for f in report.findings if "Monte Carlo" in f.message]
        self.assertTrue(len(mc_hints) >= 1)

    def test_duplicate_rule(self):
        rule = PolicyRule("total_workers", Operator.LE, 50)
        policy = SafetyPolicy("Dup", [rule, rule,
            PolicyRule("max_depth_reached", Operator.LE, 5)])
        report = self.linter.lint(policy)
        dups = [f for f in report.findings if f.category == FindingCategory.REDUNDANCY
                and "Duplicate" in f.message]
        self.assertTrue(len(dups) >= 1)

    def test_subsumed_rule(self):
        policy = SafetyPolicy("Subsumed", [
            PolicyRule("total_workers", Operator.LE, 20, Severity.ERROR),
            PolicyRule("total_workers", Operator.LE, 100, Severity.ERROR),
            PolicyRule("max_depth_reached", Operator.LE, 5),
        ])
        report = self.linter.lint(policy)
        subs = [f for f in report.findings if f.category == FindingCategory.REDUNDANCY
                and "subsumed" in f.message]
        self.assertTrue(len(subs) >= 1)

    def test_critical_metric_info_severity(self):
        policy = SafetyPolicy("BadSev", [
            PolicyRule("total_workers", Operator.LE, 50, Severity.INFO),
            PolicyRule("max_depth_reached", Operator.LE, 5),
        ])
        report = self.linter.lint(policy)
        sevs = [f for f in report.findings if f.category == FindingCategory.SEVERITY_ISSUE]
        self.assertTrue(len(sevs) >= 1)

    def test_no_denial_rate_hint(self):
        policy = SafetyPolicy("NoDenial", [
            PolicyRule("total_workers", Operator.LE, 50),
            PolicyRule("max_depth_reached", Operator.LE, 5),
        ])
        report = self.linter.lint(policy)
        hints = [f for f in report.findings if "denial_rate" in f.message]
        self.assertTrue(len(hints) >= 1)

    def test_no_descriptions_style(self):
        policy = SafetyPolicy("NoDesc", [
            PolicyRule("total_workers", Operator.LE, 50),
            PolicyRule("max_depth_reached", Operator.LE, 5),
            PolicyRule("denial_rate", Operator.GE, 0.3),
        ])
        report = self.linter.lint(policy)
        descs = [f for f in report.findings if "descriptions" in f.message.lower()]
        self.assertTrue(len(descs) >= 1)

    def test_clean_policy(self):
        """A well-formed policy should have no errors."""
        policy = SafetyPolicy("Good", [
            PolicyRule("total_workers", Operator.LE, 50, Severity.ERROR,
                       "Worker limit"),
            PolicyRule("max_depth_reached", Operator.LE, 5, Severity.ERROR,
                       "Depth limit"),
            PolicyRule("denial_rate", Operator.GE, 0.3, Severity.WARNING,
                       "Contract enforcement"),
        ])
        report = self.linter.lint(policy)
        self.assertTrue(report.passed)

    def test_builtin_presets_pass(self):
        """All built-in presets should lint without errors."""
        for name in POLICY_PRESETS:
            policy = SafetyPolicy.from_preset(name)
            report = self.linter.lint(policy)
            self.assertTrue(report.passed,
                            f"Preset '{name}' has lint errors: {report.summary()}")

    def test_mc_metric_in_non_mc_rule(self):
        """MC-only metric used in non-MC rule should be flagged."""
        policy = SafetyPolicy("BadMC", [
            PolicyRule("prob_max_depth", Operator.LE, 0.5, monte_carlo=False),
            PolicyRule("total_workers", Operator.LE, 50),
            PolicyRule("max_depth_reached", Operator.LE, 5),
        ])
        report = self.linter.lint(policy)
        invalids = [f for f in report.findings if f.category == FindingCategory.INVALID_METRIC]
        self.assertTrue(len(invalids) >= 1)

    def test_render_output(self):
        policy = SafetyPolicy("RenderTest", [
            PolicyRule("total_workers", Operator.LE, 50),
            PolicyRule("max_depth_reached", Operator.LE, 5),
        ])
        report = self.linter.lint(policy)
        text = report.render()
        self.assertIn("Policy Lint Report", text)
        self.assertIn("RenderTest", text)

    def test_json_output(self):
        policy = SafetyPolicy("JSON", [
            PolicyRule("total_workers", Operator.LE, 50),
            PolicyRule("max_depth_reached", Operator.LE, 5),
        ])
        report = self.linter.lint(policy)
        d = report.to_dict()
        parsed = json.loads(json.dumps(d))  # Ensure serializable
        self.assertEqual(parsed["policy_name"], "JSON")

if __name__ == "__main__":
    unittest.main()
