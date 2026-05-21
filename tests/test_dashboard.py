"""Tests for dashboard module."""

import json
import pytest

from replication.dashboard import (
    DashboardConfig,
    DashboardGenerator,
    _depth_color,
    _esc,
    _event_badge,
)
from replication.simulator import ScenarioConfig, Simulator, SimulationReport


# ── Fixtures ─────────────────────────────────────────────────────

@pytest.fixture
def report():
    sim = Simulator(ScenarioConfig(strategy="greedy", max_depth=3, seed=42))
    return sim.run()


@pytest.fixture
def reports():
    result = []
    for strat in ["greedy", "random", "conservative"]:
        sim = Simulator(ScenarioConfig(strategy=strat, max_depth=3, seed=42))
        result.append(sim.run())
    return result


@pytest.fixture
def gen():
    return DashboardGenerator()


# ── DashboardConfig ──────────────────────────────────────────────

class TestDashboardConfig:
    def test_defaults(self):
        cfg = DashboardConfig()
        assert cfg.title == "Replication Safety Dashboard"
        assert cfg.theme == "light"
        assert cfg.show_timeline is True
        assert cfg.show_tree is True
        assert cfg.show_audit is True
        assert cfg.max_timeline_events == 200
        assert cfg.max_audit_events == 100

    def test_custom(self):
        cfg = DashboardConfig(title="Test", theme="dark", show_tree=False)
        assert cfg.title == "Test"
        assert cfg.theme == "dark"
        assert cfg.show_tree is False


# ── Helpers ──────────────────────────────────────────────────────

class TestEsc:
    def test_escapes_html(self):
        assert _esc("<script>alert('xss')</script>") == "&lt;script&gt;alert(&#x27;xss&#x27;)&lt;/script&gt;"

    def test_plain_text(self):
        assert _esc("hello") == "hello"

    def test_ampersand(self):
        assert "&amp;" in _esc("a & b")


class TestDepthColor:
    def test_returns_string(self):
        for i in range(15):
            color = _depth_color(i)
            assert color.startswith("#")
            assert len(color) == 7

    def test_wraps_around(self):
        assert _depth_color(0) == _depth_color(10)


class TestEventBadge:
    def test_denied(self):
        assert _event_badge("replication_denied") == "badge-red"

    def test_success(self):
        assert _event_badge("task_completed") == "badge-green"

    def test_warning(self):
        assert _event_badge("rate_limit") == "badge-yellow"

    def test_default(self):
        assert _event_badge("unknown_event") == "badge-blue"


# ── Single Report ────────────────────────────────────────────────

class TestSingleReport:
    def test_generates_html(self, gen, report):
        html = gen.single_report(report)
        assert "<!DOCTYPE html>" in html
        assert "</html>" in html

    def test_contains_title(self, gen, report):
        html = gen.single_report(report, title="My Test Run")
        assert "My Test Run" in html

    def test_contains_strategy(self, gen, report):
        html = gen.single_report(report)
        assert "greedy" in html.lower()

    def test_contains_summary_cards(self, gen, report):
        html = gen.single_report(report)
        assert "card-value" in html
        assert "Workers" in html
        assert "Tasks" in html
        assert "Replications" in html

    def test_contains_depth_chart(self, gen, report):
        html = gen.single_report(report)
        assert "bar-fill" in html
        assert "Depth 0" in html

    def test_contains_tree(self, gen, report):
        html = gen.single_report(report)
        assert "Worker Lineage" in html

    def test_contains_config(self, gen, report):
        html = gen.single_report(report)
        assert "Configuration" in html
        assert "Strategy" in html

    def test_contains_footer(self, gen, report):
        html = gen.single_report(report)
        assert "replication.dashboard" in html

    def test_contains_css(self, gen, report):
        html = gen.single_report(report)
        assert "<style>" in html
        assert "card-bg" in html

    def test_contains_js(self, gen, report):
        html = gen.single_report(report)
        assert "<script>" in html

    def test_no_tree_option(self, report):
        gen = DashboardGenerator(DashboardConfig(show_tree=False))
        html = gen.single_report(report)
        assert "Worker Lineage" not in html

    def test_no_timeline_option(self, report):
        gen = DashboardGenerator(DashboardConfig(show_timeline=False))
        html = gen.single_report(report)
        assert "<h2>Timeline Events</h2>" not in html

    def test_dark_theme(self, report):
        gen = DashboardGenerator(DashboardConfig(theme="dark"))
        html = gen.single_report(report)
        assert 'data-theme="dark"' in html

    def test_timeline_truncation(self, report):
        gen = DashboardGenerator(DashboardConfig(max_timeline_events=2))
        html = gen.single_report(report)
        if len(report.timeline) > 2:
            assert "Showing 2 of" in html


# ── Comparative Report ───────────────────────────────────────────

class TestCompareReports:
    def test_generates_html(self, gen, reports):
        html = gen.compare_reports(reports, ["A", "B", "C"])
        assert "<!DOCTYPE html>" in html
        assert "Comparison" in html

    def test_contains_comparison_table(self, gen, reports):
        html = gen.compare_reports(reports, ["A", "B", "C"])
        assert "Side-by-Side" in html
        assert "<table>" in html

    def test_contains_labels(self, gen, reports):
        html = gen.compare_reports(reports, ["Alpha", "Beta", "Gamma"])
        assert "Alpha" in html
        assert "Beta" in html
        assert "Gamma" in html

    def test_contains_bars(self, gen, reports):
        html = gen.compare_reports(reports, ["A", "B", "C"])
        assert "Worker Count Comparison" in html

    def test_contains_tabs(self, gen, reports):
        html = gen.compare_reports(reports, ["A", "B", "C"])
        assert "tab-btn" in html
        assert "tab-panel" in html

    def test_default_labels(self, gen, reports):
        html = gen.compare_reports(reports)
        assert "Run 1" in html
        assert "greedy" in html.lower()

    def test_empty_reports(self, gen):
        html = gen.compare_reports([])
        assert "No simulation data" in html

    def test_single_report_in_compare(self, gen, reports):
        html = gen.compare_reports(reports[:1], ["Solo"])
        assert "Solo" in html


# ── Report Data ──────────────────────────────────────────────────

class TestGetReportData:
    def test_returns_dict(self, gen, report):
        data = gen.get_report_data(report)
        assert isinstance(data, dict)

    def test_has_expected_keys(self, gen, report):
        data = gen.get_report_data(report)
        expected = {
            "strategy", "worker_count", "max_depth", "total_tasks",
            "replications_attempted", "replications_succeeded",
            "replications_denied", "success_rate", "duration_ms",
            "depth_distribution", "timeline_count", "audit_count",
        }
        assert set(data.keys()) == expected

    def test_strategy_matches(self, gen, report):
        data = gen.get_report_data(report)
        assert data["strategy"] == "greedy"

    def test_worker_count_positive(self, gen, report):
        data = gen.get_report_data(report)
        assert data["worker_count"] > 0

    def test_success_rate_range(self, gen, report):
        data = gen.get_report_data(report)
        assert 0 <= data["success_rate"] <= 1.0

    def test_depth_distribution_dict(self, gen, report):
        data = gen.get_report_data(report)
        assert isinstance(data["depth_distribution"], dict)
        assert 0 in data["depth_distribution"]

    def test_json_serializable(self, gen, report):
        data = gen.get_report_data(report)
        json.dumps(data)


# ── Determinism ──────────────────────────────────────────────────

class TestDeterminism:
    def test_same_seed_same_html(self, gen):
        sim1 = Simulator(ScenarioConfig(strategy="greedy", seed=42))
        sim2 = Simulator(ScenarioConfig(strategy="greedy", seed=42))
        r1 = sim1.run()
        r2 = sim2.run()
        # Reports should have the same logical data for the same seed.
        # ``duration_ms`` reflects wall-clock execution time and is
        # intentionally non-deterministic, so we use the deterministic
        # view here (mirroring how HTML output's timestamp is excluded).
        d1 = gen.get_report_data_deterministic(r1)
        d2 = gen.get_report_data_deterministic(r2)
        assert d1 == d2
        # ``duration_ms`` must not leak into the deterministic view.
        assert "duration_ms" not in d1
        # But the full view still reports it.
        assert "duration_ms" in gen.get_report_data(r1)

    def test_include_nondeterministic_flag(self, gen):
        sim = Simulator(ScenarioConfig(strategy="greedy", seed=7))
        report = sim.run()
        full = gen.get_report_data(report, include_nondeterministic=True)
        lean = gen.get_report_data(report, include_nondeterministic=False)
        # Lean view drops every documented non-deterministic key, and
        # nothing else, so the two views differ only by that set.
        assert set(full) - set(lean) == set(gen.NONDETERMINISTIC_KEYS)
        for key in gen.NONDETERMINISTIC_KEYS:
            assert key not in lean
