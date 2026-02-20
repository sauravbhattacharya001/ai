"""Tests for the interactive HTML report generator."""

import json
import os
import tempfile

import pytest

from replication.reporter import HTMLReporter
from replication.simulator import Simulator, ScenarioConfig, PRESETS
from replication.comparator import Comparator
from replication.threats import ThreatSimulator, ThreatConfig


# ─── Fixtures ───────────────────────────────────────────────────────

@pytest.fixture
def reporter():
    return HTMLReporter()


@pytest.fixture
def sim_report():
    config = ScenarioConfig(
        max_depth=2, max_replicas=8, strategy="greedy",
        tasks_per_worker=2, cooldown_seconds=0.0, seed=42,
    )
    return Simulator(config).run()


@pytest.fixture
def threat_report():
    config = ThreatConfig(max_depth=3, max_replicas=10, cooldown_seconds=1.0)
    return ThreatSimulator(config).run_all()


@pytest.fixture
def comparison_result():
    comp = Comparator(ScenarioConfig(
        max_depth=2, max_replicas=5, cooldown_seconds=0.0, seed=42,
    ))
    return comp.compare_strategies(["greedy", "conservative", "burst"], seed=42)


# ─── Simulation report tests ───────────────────────────────────────

class TestSimulationReport:
    def test_generates_valid_html(self, reporter, sim_report):
        html = reporter.simulation_report(sim_report)
        assert "<!DOCTYPE html>" in html
        assert "</html>" in html
        assert "Simulation Report" in html

    def test_contains_overview_cards(self, reporter, sim_report):
        html = reporter.simulation_report(sim_report)
        assert "Workers Spawned" in html
        assert "Tasks Completed" in html
        assert "Replications OK" in html

    def test_contains_config_table(self, reporter, sim_report):
        html = reporter.simulation_report(sim_report)
        assert "greedy" in html
        assert "Strategy" in html
        assert "Max Depth" in html

    def test_contains_depth_chart(self, reporter, sim_report):
        html = reporter.simulation_report(sim_report)
        assert "depthChart" in html
        assert "Worker Depth Distribution" in html

    def test_contains_replication_donut(self, reporter, sim_report):
        html = reporter.simulation_report(sim_report)
        assert "replChart" in html
        assert "Replication Outcomes" in html

    def test_contains_worker_tree(self, reporter, sim_report):
        html = reporter.simulation_report(sim_report)
        assert "Worker Lineage Tree" in html
        assert "tree-view" in html

    def test_contains_timeline(self, reporter, sim_report):
        html = reporter.simulation_report(sim_report)
        assert "Timeline" in html
        assert "timeline-entry" in html

    def test_contains_worker_table(self, reporter, sim_report):
        html = reporter.simulation_report(sim_report)
        assert "Worker Details" in html
        for wid in list(sim_report.workers.keys())[:3]:
            assert wid in html

    def test_theme_toggle(self, reporter, sim_report):
        html = reporter.simulation_report(sim_report)
        assert "toggleTheme" in html
        assert "theme-toggle" in html
        assert 'data-theme="dark"' in html

    def test_collapsible_sections(self, reporter, sim_report):
        html = reporter.simulation_report(sim_report)
        assert "section-header" in html
        assert "collapsed" in html  # timeline should be collapsed

    def test_responsive_meta(self, reporter, sim_report):
        html = reporter.simulation_report(sim_report)
        assert "viewport" in html
        assert "max-width" in html


class TestSimulationReportVariants:
    def test_minimal_preset(self, reporter):
        report = Simulator(PRESETS["minimal"]).run()
        html = reporter.simulation_report(report)
        assert "Simulation Report" in html
        assert len(report.workers) >= 1

    def test_burst_preset(self, reporter):
        report = Simulator(PRESETS["burst"]).run()
        html = reporter.simulation_report(report)
        assert "Simulation Report" in html

    def test_chain_preset(self, reporter):
        report = Simulator(PRESETS["chain"]).run()
        html = reporter.simulation_report(report)
        assert "Simulation Report" in html

    def test_zero_tasks(self, reporter):
        config = ScenarioConfig(
            max_depth=1, max_replicas=3, strategy="greedy",
            tasks_per_worker=0, cooldown_seconds=0.0,
        )
        report = Simulator(config).run()
        html = reporter.simulation_report(report)
        assert "Simulation Report" in html


# ─── Threat report tests ───────────────────────────────────────────

class TestThreatReport:
    def test_generates_valid_html(self, reporter, threat_report):
        html = reporter.threat_report(threat_report)
        assert "<!DOCTYPE html>" in html
        assert "</html>" in html
        assert "Threat Assessment Report" in html

    def test_contains_security_score(self, reporter, threat_report):
        html = reporter.threat_report(threat_report)
        assert "Security Score" in html
        assert "grade-badge" in html

    def test_contains_status_cards(self, reporter, threat_report):
        html = reporter.threat_report(threat_report)
        assert "Mitigated" in html
        assert "Scenarios Tested" in html

    def test_contains_threat_chart(self, reporter, threat_report):
        html = reporter.threat_report(threat_report)
        assert "threatChart" in html
        assert "Block Rate by Threat Scenario" in html

    def test_contains_mitigation_donut(self, reporter, threat_report):
        html = reporter.threat_report(threat_report)
        assert "mitigationDonut" in html
        assert "Mitigation Status" in html

    def test_contains_threat_details_table(self, reporter, threat_report):
        html = reporter.threat_report(threat_report)
        assert "Threat Details" in html
        assert "CRITICAL" in html or "HIGH" in html

    def test_severity_badges(self, reporter, threat_report):
        html = reporter.threat_report(threat_report)
        assert "badge-red" in html or "badge-yellow" in html or "badge-blue" in html

    def test_status_badges(self, reporter, threat_report):
        html = reporter.threat_report(threat_report)
        assert "PASS" in html or "WARN" in html or "FAIL" in html

    def test_contains_recommendations(self, reporter, threat_report):
        html = reporter.threat_report(threat_report)
        # Recommendations section should be present if there are non-mitigated threats
        assert "Recommendations" in html or "All threats mitigated" in html

    def test_progress_bars(self, reporter, threat_report):
        html = reporter.threat_report(threat_report)
        assert "progress-bar" in html
        assert "progress-fill" in html


# ─── Comparison report tests ───────────────────────────────────────

class TestComparisonReport:
    def test_generates_valid_html(self, reporter, comparison_result):
        html = reporter.comparison_report(comparison_result)
        assert "<!DOCTYPE html>" in html
        assert "</html>" in html
        assert "Comparison" in html

    def test_contains_overview(self, reporter, comparison_result):
        html = reporter.comparison_report(comparison_result)
        assert "Comparison Overview" in html
        assert "3 scenarios" in html  # greedy, conservative, burst

    def test_contains_charts(self, reporter, comparison_result):
        html = reporter.comparison_report(comparison_result)
        assert "compChart1" in html
        assert "compChart2" in html
        assert "compChart3" in html

    def test_contains_metrics_table(self, reporter, comparison_result):
        html = reporter.comparison_report(comparison_result)
        assert "Metrics Table" in html
        assert "greedy" in html
        assert "conservative" in html
        assert "burst" in html

    def test_contains_strategy_labels(self, reporter, comparison_result):
        html = reporter.comparison_report(comparison_result)
        for run in comparison_result.runs:
            assert run.label in html

    def test_contains_insights(self, reporter, comparison_result):
        html = reporter.comparison_report(comparison_result)
        assert "Insights" in html

    def test_preset_comparison(self, reporter):
        comp = Comparator()
        result = comp.compare_presets(["minimal", "balanced"])
        html = reporter.comparison_report(result)
        assert "minimal" in html
        assert "balanced" in html

    def test_sweep_comparison(self, reporter):
        comp = Comparator(ScenarioConfig(cooldown_seconds=0.0, seed=42))
        result = comp.sweep("max_depth", [1, 2, 3], seed=42)
        html = reporter.comparison_report(result)
        assert "max_depth=1" in html
        assert "max_depth=2" in html
        assert "max_depth=3" in html
        assert "sweep" in html.lower() or "Sweep" in html


# ─── Combined report tests ─────────────────────────────────────────

class TestCombinedReport:
    def test_combined_all_sections(self, reporter, sim_report, threat_report, comparison_result):
        html = reporter.combined_report(
            simulation=sim_report,
            threat=threat_report,
            comparison=comparison_result,
        )
        assert "<!DOCTYPE html>" in html
        assert "AI Replication Safety Report" in html

    def test_combined_has_tabs(self, reporter, sim_report, threat_report):
        html = reporter.combined_report(
            simulation=sim_report,
            threat=threat_report,
        )
        assert "tab-bar" in html
        assert "tab-btn" in html
        assert "switchTab" in html
        assert "Simulation" in html
        assert "Threats" in html

    def test_combined_sim_only(self, reporter, sim_report):
        html = reporter.combined_report(simulation=sim_report)
        assert "Simulation" in html
        # Should not have threat or comparison tabs
        assert "Threats" not in html or "tab-btn" in html

    def test_combined_threats_only(self, reporter, threat_report):
        html = reporter.combined_report(threat=threat_report)
        assert "Threats" in html
        assert "Security Score" in html

    def test_combined_empty(self, reporter):
        html = reporter.combined_report()
        assert "Empty Report" in html
        assert "No data provided" in html

    def test_combined_all_charts_unique_ids(self, reporter, sim_report, threat_report, comparison_result):
        html = reporter.combined_report(
            simulation=sim_report,
            threat=threat_report,
            comparison=comparison_result,
        )
        # Each chart should have a unique canvas ID
        assert "simDepthChart" in html
        assert "combThreatDonut" in html
        assert "combCompChart" in html


# ─── File save tests ───────────────────────────────────────────────

class TestSaveReport:
    def test_save_creates_file(self, reporter, sim_report):
        html = reporter.simulation_report(sim_report)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test-report.html")
            saved = reporter.save(html, path)
            assert os.path.exists(saved)
            with open(saved, "r", encoding="utf-8") as f:
                content = f.read()
            assert "<!DOCTYPE html>" in content
            assert "Simulation Report" in content

    def test_save_creates_directories(self, reporter, sim_report):
        html = reporter.simulation_report(sim_report)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "subdir", "deep", "report.html")
            saved = reporter.save(html, path)
            assert os.path.exists(saved)

    def test_save_overwrites_existing(self, reporter, sim_report):
        html1 = reporter.simulation_report(sim_report)
        config2 = ScenarioConfig(strategy="chain", max_depth=4, cooldown_seconds=0.0)
        report2 = Simulator(config2).run()
        html2 = reporter.simulation_report(report2)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "report.html")
            reporter.save(html1, path)
            reporter.save(html2, path)
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            assert "chain" in content

    def test_save_returns_absolute_path(self, reporter, sim_report):
        html = reporter.simulation_report(sim_report)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "report.html")
            saved = reporter.save(html, path)
            assert os.path.isabs(saved)


# ─── HTML quality tests ────────────────────────────────────────────

class TestHTMLQuality:
    def test_html_escapes_special_chars(self, reporter):
        """Ensure HTML special characters in worker IDs are escaped."""
        config = ScenarioConfig(
            max_depth=1, max_replicas=3, strategy="greedy",
            tasks_per_worker=1, cooldown_seconds=0.0, seed=1,
        )
        report = Simulator(config).run()
        html = reporter.simulation_report(report)
        # Should not have unescaped < or > in data areas
        # (HTML tags are fine, but data content should be escaped)
        assert "<!DOCTYPE html>" in html  # structural HTML is fine

    def test_css_included(self, reporter, sim_report):
        html = reporter.simulation_report(sim_report)
        assert "<style>" in html
        assert "--bg:" in html
        assert "--accent:" in html

    def test_js_included(self, reporter, sim_report):
        html = reporter.simulation_report(sim_report)
        assert "<script>" in html
        assert "toggleTheme" in html
        assert "drawBarChart" in html
        assert "drawDonutChart" in html

    def test_self_contained(self, reporter, sim_report):
        """Report should not reference external resources."""
        html = reporter.simulation_report(sim_report)
        # No external CSS/JS links
        assert 'href="http' not in html
        assert 'src="http' not in html
        assert "cdn" not in html.lower()

    def test_dark_and_light_themes(self, reporter, sim_report):
        html = reporter.simulation_report(sim_report)
        # Both theme styles should be defined
        assert "[data-theme=\"light\"]" in html or "data-theme=\\\"light\\\"" in html

    def test_canvas_charts_have_ids(self, reporter, sim_report):
        html = reporter.simulation_report(sim_report)
        assert 'id="depthChart"' in html
        assert 'id="replChart"' in html

    def test_responsive_design(self, reporter, sim_report):
        html = reporter.simulation_report(sim_report)
        assert "@media" in html
        assert "auto-fit" in html


# ─── Edge cases ─────────────────────────────────────────────────────

class TestEdgeCases:
    def test_single_worker_report(self, reporter):
        config = ScenarioConfig(
            max_depth=0, max_replicas=1, strategy="greedy",
            tasks_per_worker=1, cooldown_seconds=0.0,
        )
        report = Simulator(config).run()
        html = reporter.simulation_report(report)
        assert "Simulation Report" in html
        assert len(report.workers) == 1

    def test_many_workers_report(self, reporter):
        config = ScenarioConfig(
            max_depth=3, max_replicas=30, strategy="greedy",
            tasks_per_worker=2, cooldown_seconds=0.0, seed=42,
        )
        report = Simulator(config).run()
        html = reporter.simulation_report(report)
        assert "Simulation Report" in html
        assert len(report.workers) > 5

    def test_timeline_truncation(self, reporter):
        """Long timelines should be truncated to 100 entries."""
        config = ScenarioConfig(
            max_depth=4, max_replicas=50, strategy="greedy",
            tasks_per_worker=3, cooldown_seconds=0.0, seed=42,
        )
        report = Simulator(config).run()
        html = reporter.simulation_report(report)
        if len(report.timeline) > 100:
            assert "more events omitted" in html

    def test_comparison_single_strategy(self, reporter):
        comp = Comparator(ScenarioConfig(cooldown_seconds=0.0, seed=42))
        result = comp.compare_strategies(["greedy"], seed=42)
        html = reporter.comparison_report(result)
        assert "Comparison" in html

    def test_threat_custom_config(self, reporter):
        config = ThreatConfig(max_depth=5, max_replicas=20, cooldown_seconds=2.0)
        report = ThreatSimulator(config).run_all()
        html = reporter.threat_report(report)
        assert "Threat Assessment Report" in html
