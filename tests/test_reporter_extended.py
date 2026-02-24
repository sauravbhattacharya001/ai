"""Extended tests for HTMLReporter — edge cases, CSS, empty/extreme inputs, HTML structure."""

import json
import os
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pytest

from replication.reporter import HTMLReporter
from replication.simulator import (
    Simulator,
    SimulationReport,
    ScenarioConfig,
    WorkerRecord,
    PRESETS,
)
from replication.comparator import Comparator, ComparisonResult
from replication.threats import (
    ThreatSimulator,
    ThreatConfig,
    ThreatReport,
    ThreatResult,
    MitigationStatus,
    ThreatSeverity,
)


# ─── Fixtures ───────────────────────────────────────────────────────


@pytest.fixture
def reporter():
    return HTMLReporter()


def _make_sim_report(
    *,
    workers: Optional[Dict[str, WorkerRecord]] = None,
    config: Optional[ScenarioConfig] = None,
    timeline: Optional[List[Dict[str, Any]]] = None,
) -> SimulationReport:
    """Build a SimulationReport from scratch for testing."""
    if config is None:
        config = ScenarioConfig(
            max_depth=2, max_replicas=5, strategy="greedy",
            tasks_per_worker=1, cooldown_seconds=0.0,
        )
    if workers is None:
        workers = {
            "root": WorkerRecord(
                worker_id="root", parent_id=None, depth=0,
                tasks_completed=1, replications_succeeded=0,
                replications_denied=0, children=[],
            ),
        }
    root_id = next(
        (wid for wid, w in workers.items() if w.parent_id is None), "root"
    )
    total_tasks = sum(w.tasks_completed for w in workers.values())
    total_ok = sum(w.replications_succeeded for w in workers.values())
    total_denied = sum(w.replications_denied for w in workers.values())
    return SimulationReport(
        config=config,
        workers=workers,
        root_id=root_id,
        timeline=timeline or [],
        total_tasks=total_tasks,
        total_replications_attempted=total_ok + total_denied,
        total_replications_succeeded=total_ok,
        total_replications_denied=total_denied,
        duration_ms=0.0,
        audit_events=[],
    )


def _make_threat_report(results: Optional[List[ThreatResult]] = None) -> ThreatReport:
    """Build a ThreatReport from explicit results."""
    config = ThreatConfig(max_depth=2, max_replicas=5, cooldown_seconds=0.0)
    if results is None:
        results = []
    return ThreatReport(config=config, results=results, duration_ms=0.0)


# ─── Empty / Minimal inputs ────────────────────────────────────────


class TestEmptyInputs:
    """Reports generated from empty or minimal data should not crash."""

    def test_simulation_no_workers(self, reporter):
        """A report with zero workers (edge: root only, no children)."""
        report = _make_sim_report()
        html = reporter.simulation_report(report)
        assert "<!DOCTYPE html>" in html
        assert "Simulation Report" in html

    def test_simulation_empty_timeline(self, reporter):
        report = _make_sim_report(timeline=[])
        html = reporter.simulation_report(report)
        assert "Timeline" in html

    def test_threat_no_results(self, reporter):
        """ThreatReport with zero scenarios tested."""
        report = _make_threat_report([])
        html = reporter.threat_report(report)
        assert "Threat Assessment Report" in html
        # Score / grade should still render
        assert "Security Score" in html

    def test_combined_no_sections(self, reporter):
        html = reporter.combined_report()
        assert "Empty Report" in html
        assert "No data provided" in html

    def test_combined_sim_only_no_threat(self, reporter):
        report = _make_sim_report()
        html = reporter.combined_report(simulation=report)
        assert "Simulation" in html


# ─── Extreme values ────────────────────────────────────────────────


class TestExtremeValues:
    """Ensure the reporter handles large or unusual numbers gracefully."""

    def test_very_large_worker_count(self, reporter):
        workers = {}
        for i in range(200):
            workers[f"w-{i}"] = WorkerRecord(
                worker_id=f"w-{i}",
                parent_id=None if i == 0 else "w-0",
                depth=0 if i == 0 else 1,
                tasks_completed=i,
                replications_succeeded=i % 2,
                replications_denied=1 - (i % 2),
                children=[],
            )
        workers["w-0"].children = [f"w-{i}" for i in range(1, 200)]
        report = _make_sim_report(workers=workers)
        html = reporter.simulation_report(report)
        assert "Simulation Report" in html
        # All 200 workers should appear in the worker table
        assert "w-199" in html

    def test_very_deep_depth(self, reporter):
        workers = {}
        for i in range(20):
            workers[f"d-{i}"] = WorkerRecord(
                worker_id=f"d-{i}",
                parent_id=None if i == 0 else f"d-{i-1}",
                depth=i,
                tasks_completed=1,
                replications_succeeded=1 if i < 19 else 0,
                replications_denied=0,
                children=[f"d-{i+1}"] if i < 19 else [],
            )
        report = _make_sim_report(workers=workers)
        html = reporter.simulation_report(report)
        assert "Max Depth" in html
        # Depth 19 should appear
        assert "Depth 19" in html

    def test_zero_tasks_zero_replications(self, reporter):
        workers = {
            "idle": WorkerRecord(
                worker_id="idle", parent_id=None, depth=0,
                tasks_completed=0, replications_succeeded=0,
                replications_denied=0, children=[],
            ),
        }
        report = _make_sim_report(workers=workers)
        html = reporter.simulation_report(report)
        assert "Simulation Report" in html

    def test_threat_all_failed(self, reporter):
        results = [
            ThreatResult(
                scenario_id=f"fail-{i}", name=f"Fail Scenario {i}",
                description="Everything fails",
                severity=ThreatSeverity.CRITICAL,
                status=MitigationStatus.FAILED,
                attacks_attempted=100, attacks_blocked=0, attacks_succeeded=100,
                details=[f"Failed attempt {j}" for j in range(5)],
                duration_ms=1.0, audit_events=[],
            )
            for i in range(5)
        ]
        report = _make_threat_report(results)
        html = reporter.threat_report(report)
        assert "Threat Assessment Report" in html
        assert "FAIL" in html

    def test_threat_all_mitigated(self, reporter):
        results = [
            ThreatResult(
                scenario_id="pass-1", name="Good Scenario",
                description="Everything passes",
                severity=ThreatSeverity.LOW,
                status=MitigationStatus.MITIGATED,
                attacks_attempted=50, attacks_blocked=50, attacks_succeeded=0,
                details=["All blocked"],
                duration_ms=0.5, audit_events=[],
            ),
        ]
        report = _make_threat_report(results)
        html = reporter.threat_report(report)
        assert "PASS" in html

    def test_threat_zero_attacks(self, reporter):
        """A threat result with zero attacks attempted (block_rate edge)."""
        results = [
            ThreatResult(
                scenario_id="zero", name="Zero Attacks",
                description="Nothing attempted",
                severity=ThreatSeverity.MEDIUM,
                status=MitigationStatus.MITIGATED,
                attacks_attempted=0, attacks_blocked=0, attacks_succeeded=0,
                details=[],
                duration_ms=0.0, audit_events=[],
            ),
        ]
        report = _make_threat_report(results)
        html = reporter.threat_report(report)
        assert "Threat Assessment Report" in html

    def test_long_timeline_truncation(self, reporter):
        """Timeline with >100 entries should show truncation message."""
        timeline = [
            {"type": "task", "time_ms": float(i), "worker_id": "root", "detail": f"Task {i}"}
            for i in range(150)
        ]
        report = _make_sim_report(timeline=timeline)
        html = reporter.simulation_report(report)
        assert "more events omitted" in html


# ─── CSS generation ─────────────────────────────────────────────────


class TestCSSGeneration:
    """Verify the CSS output contains expected design tokens and rules."""

    def test_css_contains_root_variables(self):
        css = HTMLReporter._css()
        for var in ["--bg", "--surface", "--text", "--accent", "--danger", "--warning"]:
            assert var in css, f"Missing CSS variable {var}"

    def test_css_contains_light_theme(self):
        css = HTMLReporter._css()
        assert '[data-theme="light"]' in css

    def test_css_contains_card_styles(self):
        css = HTMLReporter._css()
        assert ".card" in css
        assert ".cards" in css

    def test_css_contains_section_styles(self):
        css = HTMLReporter._css()
        assert ".section" in css
        assert ".section-header" in css
        assert ".section-body" in css
        assert ".collapsed" in css

    def test_css_contains_badge_classes(self):
        css = HTMLReporter._css()
        for badge in [".badge-green", ".badge-red", ".badge-yellow", ".badge-blue"]:
            assert badge in css, f"Missing badge class {badge}"

    def test_css_contains_progress_bar(self):
        css = HTMLReporter._css()
        assert ".progress-bar" in css
        assert ".progress-fill" in css

    def test_css_contains_responsive_breakpoint(self):
        css = HTMLReporter._css()
        assert "@media" in css
        assert "768px" in css

    def test_css_contains_table_styles(self):
        css = HTMLReporter._css()
        assert "table" in css
        assert "border-collapse" in css

    def test_css_contains_grade_badges(self):
        css = HTMLReporter._css()
        for grade in [".grade-a", ".grade-b", ".grade-c", ".grade-d"]:
            assert grade in css, f"Missing grade class {grade}"

    def test_css_contains_tab_styles(self):
        css = HTMLReporter._css()
        assert ".tab-bar" in css
        assert ".tab-btn" in css
        assert ".tab-panel" in css


# ─── JavaScript generation ──────────────────────────────────────────


class TestJSGeneration:
    """Verify the JavaScript output contains expected functions."""

    def test_js_has_toggle_theme(self):
        js = HTMLReporter._js()
        assert "function toggleTheme()" in js

    def test_js_has_switch_tab(self):
        js = HTMLReporter._js()
        assert "function switchTab(" in js

    def test_js_has_draw_bar_chart(self):
        js = HTMLReporter._js()
        assert "function drawBarChart(" in js

    def test_js_has_draw_donut_chart(self):
        js = HTMLReporter._js()
        assert "function drawDonutChart(" in js

    def test_js_has_resize_handler(self):
        js = HTMLReporter._js()
        assert "resize" in js

    def test_js_has_palette(self):
        js = HTMLReporter._js()
        assert "PALETTE" in js

    def test_js_has_collapsible_sections(self):
        js = HTMLReporter._js()
        assert "section-header" in js

    def test_js_has_theme_colors_function(self):
        js = HTMLReporter._js()
        assert "function getThemeColors()" in js


# ─── HTML structure verification ────────────────────────────────────


class TestHTMLStructure:
    """Verify structural HTML elements are correct in generated reports."""

    def test_simulation_html_structure(self, reporter):
        report = Simulator(ScenarioConfig(
            max_depth=2, max_replicas=5, strategy="greedy",
            tasks_per_worker=2, cooldown_seconds=0.0, seed=42,
        )).run()
        html = reporter.simulation_report(report)

        # Valid HTML5 document
        assert html.startswith("<!DOCTYPE html>")
        assert "<html" in html
        assert "</html>" in html
        assert "<head>" in html
        assert "</head>" in html
        assert "<body>" in html
        assert "</body>" in html

        # Meta tags
        assert 'charset="UTF-8"' in html
        assert "viewport" in html

        # Has header with title and theme toggle
        assert "<header>" in html
        assert "theme-toggle" in html

        # Has script and style tags
        assert "<style>" in html
        assert "<script>" in html

    def test_threat_html_has_grade_badge(self, reporter):
        config = ThreatConfig(max_depth=2, max_replicas=5, cooldown_seconds=0.0)
        report = ThreatSimulator(config).run_all()
        html = reporter.threat_report(report)
        assert "grade-badge" in html

    def test_comparison_html_has_metric_cards(self, reporter):
        comp = Comparator(ScenarioConfig(
            max_depth=2, max_replicas=5, cooldown_seconds=0.0, seed=42,
        ))
        result = comp.compare_strategies(["greedy", "conservative"], seed=42)
        html = reporter.comparison_report(result)
        assert "Scenarios" in html
        assert "Most Workers" in html

    def test_combined_html_has_tab_structure(self, reporter):
        sim = Simulator(ScenarioConfig(
            max_depth=1, max_replicas=3, cooldown_seconds=0.0, seed=42,
        )).run()
        threat = ThreatSimulator(ThreatConfig(
            max_depth=2, max_replicas=5, cooldown_seconds=0.0,
        )).run_all()
        html = reporter.combined_report(simulation=sim, threat=threat)
        assert 'data-tab-group="main"' in html
        assert 'tab-bar' in html
        assert 'tab-panel' in html


# ─── Helper methods ─────────────────────────────────────────────────


class TestHelperMethods:
    """Test individual helper methods on HTMLReporter."""

    def test_esc_html_entities(self, reporter):
        assert reporter._esc("<script>") == "&lt;script&gt;"
        assert reporter._esc('"quotes"') == "&quot;quotes&quot;"
        assert reporter._esc("a & b") == "a &amp; b"
        assert reporter._esc("safe text") == "safe text"

    def test_esc_non_string_input(self, reporter):
        assert reporter._esc(42) == "42"
        assert reporter._esc(3.14) == "3.14"
        assert reporter._esc(None) == "None"

    def test_badge_returns_html(self, reporter):
        html = reporter._badge("OK", "green")
        assert "badge-green" in html
        assert "OK" in html

    def test_badge_kinds(self, reporter):
        for kind in ["green", "red", "yellow", "blue"]:
            html = reporter._badge("test", kind)
            assert f"badge-{kind}" in html

    def test_progress_bar_clamping(self, reporter):
        # Values > 100 should be clamped
        html = reporter._progress(150.0)
        assert 'width:100.0%' in html

        # Values < 0 should be clamped
        html = reporter._progress(-10.0)
        assert 'width:0.0%' in html

    def test_progress_bar_normal(self, reporter):
        html = reporter._progress(75.0)
        assert 'width:75.0%' in html
        assert "75%" in html

    def test_progress_bar_custom_label(self, reporter):
        html = reporter._progress(50.0, label="Half")
        assert "Half" in html

    def test_progress_bar_custom_color(self, reporter):
        html = reporter._progress(60.0, color="#ff0000")
        assert "#ff0000" in html

    def test_card_returns_html(self, reporter):
        html = reporter._card("42", "Workers")
        assert "42" in html
        assert "Workers" in html
        assert "card" in html

    def test_card_with_color(self, reporter):
        html = reporter._card("100", "Score", color="#3fb950")
        assert "#3fb950" in html

    def test_section_collapsed(self, reporter):
        html = reporter._section("Title", "<p>Body</p>", collapsed=True)
        assert "collapsed" in html

    def test_section_with_icon(self, reporter):
        html = reporter._section("Title", "<p>Body</p>", icon="🔥")
        assert "🔥" in html

    def test_section_not_collapsed_by_default(self, reporter):
        html = reporter._section("Title", "<p>Body</p>")
        # Should not have the collapsed class in the opening div
        assert 'class="section">' in html

    def test_wrap_page_structure(self, reporter):
        html = reporter._wrap_page("Test Title", "<p>Content</p>")
        assert "Test Title" in html
        assert "<p>Content</p>" in html
        assert "<!DOCTYPE html>" in html
        assert 'data-theme="dark"' in html


# ─── Serialization / to_dict ────────────────────────────────────────


class TestSerialization:
    """Verify report data can be serialized to JSON (via to_dict)."""

    def test_simulation_report_to_dict(self):
        config = ScenarioConfig(
            max_depth=2, max_replicas=5, strategy="greedy",
            tasks_per_worker=1, cooldown_seconds=0.0, seed=42,
        )
        report = Simulator(config).run()
        data = report.to_dict()
        assert isinstance(data, dict)
        # Should be JSON-serializable
        json_str = json.dumps(data)
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert "workers" in parsed
        assert "config" in parsed

    def test_threat_report_to_dict(self):
        config = ThreatConfig(max_depth=2, max_replicas=5, cooldown_seconds=0.0)
        report = ThreatSimulator(config).run_all()
        data = report.to_dict()
        assert isinstance(data, dict)
        json_str = json.dumps(data)
        parsed = json.loads(json_str)
        assert "results" in parsed

    def test_simulation_render_methods(self):
        config = ScenarioConfig(
            max_depth=2, max_replicas=5, strategy="greedy",
            tasks_per_worker=1, cooldown_seconds=0.0, seed=42,
        )
        report = Simulator(config).run()
        assert isinstance(report.render(), str)
        assert isinstance(report.render_summary(), str)
        assert isinstance(report.render_timeline(), str)
        assert isinstance(report.render_tree(), str)

    def test_threat_render_methods(self):
        config = ThreatConfig(max_depth=2, max_replicas=5, cooldown_seconds=0.0)
        report = ThreatSimulator(config).run_all()
        assert isinstance(report.render(), str)
        assert isinstance(report.render_summary(), str)
        assert isinstance(report.render_details(), str)
        assert isinstance(report.render_matrix(), str)
        assert isinstance(report.render_recommendations(), str)


# ─── Threat result edge cases ───────────────────────────────────────


class TestThreatResultEdgeCases:
    """Edge cases for ThreatResult block_rate and badge rendering."""

    def test_block_rate_100_percent(self):
        r = ThreatResult(
            "t1", "Test", "D", ThreatSeverity.HIGH, MitigationStatus.MITIGATED,
            attacks_attempted=100, attacks_blocked=100, attacks_succeeded=0,
            details=[], duration_ms=0.0, audit_events=[],
        )
        assert r.block_rate == 100.0

    def test_block_rate_0_percent(self):
        r = ThreatResult(
            "t2", "Test", "D", ThreatSeverity.CRITICAL, MitigationStatus.FAILED,
            attacks_attempted=100, attacks_blocked=0, attacks_succeeded=100,
            details=[], duration_ms=0.0, audit_events=[],
        )
        assert r.block_rate == 0.0

    def test_block_rate_zero_attacks(self):
        """block_rate when attacks_attempted is zero."""
        r = ThreatResult(
            "t3", "Test", "D", ThreatSeverity.LOW, MitigationStatus.MITIGATED,
            attacks_attempted=0, attacks_blocked=0, attacks_succeeded=0,
            details=[], duration_ms=0.0, audit_events=[],
        )
        # Should not raise ZeroDivisionError
        br = r.block_rate
        assert isinstance(br, (int, float))

    def test_all_severity_levels_render(self, reporter):
        """Ensure all four severity levels render without errors."""
        results = [
            ThreatResult(
                f"s-{sev.name}", f"Threat {sev.name}", f"Desc {sev.name}",
                sev, MitigationStatus.PARTIAL,
                attacks_attempted=10, attacks_blocked=5, attacks_succeeded=5,
                details=["detail"], duration_ms=0.1, audit_events=[],
            )
            for sev in ThreatSeverity
        ]
        report = _make_threat_report(results)
        html = reporter.threat_report(report)
        for sev in ThreatSeverity:
            assert sev.name in html.upper() or sev.name.lower() in html.lower()

    def test_all_mitigation_statuses_render(self, reporter):
        results = [
            ThreatResult(
                f"m-{st.name}", f"Threat {st.name}", f"Desc",
                ThreatSeverity.MEDIUM, st,
                attacks_attempted=10, attacks_blocked=5, attacks_succeeded=5,
                details=["d1"], duration_ms=0.1, audit_events=[],
            )
            for st in MitigationStatus
        ]
        report = _make_threat_report(results)
        html = reporter.threat_report(report)
        assert "PASS" in html
        assert "WARN" in html
        assert "FAIL" in html

    def test_threat_details_truncation_in_table(self, reporter):
        """Details longer than 3 should show +N more."""
        results = [
            ThreatResult(
                "detail-test", "Many Details", "D",
                ThreatSeverity.HIGH, MitigationStatus.PARTIAL,
                attacks_attempted=10, attacks_blocked=5, attacks_succeeded=5,
                details=[f"Detail line {i}" for i in range(10)],
                duration_ms=0.1, audit_events=[],
            ),
        ]
        report = _make_threat_report(results)
        html = reporter.threat_report(report)
        assert "+7 more" in html


# ─── File operations extended ───────────────────────────────────────


class TestFileOperationsExtended:
    def test_save_utf8_content(self, reporter):
        """Ensure UTF-8 characters in reports are preserved."""
        workers = {
            "emoji-worker": WorkerRecord(
                worker_id="emoji-worker", parent_id=None, depth=0,
                tasks_completed=1, children=[],
            ),
        }
        report = _make_sim_report(workers=workers)
        html = reporter.simulation_report(report)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "utf8-report.html")
            saved = reporter.save(html, path)
            with open(saved, "r", encoding="utf-8") as f:
                content = f.read()
            # Should contain emoji from section icons
            assert "🧬" in content or "📊" in content

    def test_save_empty_html(self, reporter):
        """save() should handle an empty string."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "empty.html")
            saved = reporter.save("", path)
            assert os.path.exists(saved)
            assert os.path.getsize(saved) == 0


# ─── Multiple report types in sequence ──────────────────────────────


class TestMultipleReportTypes:
    """Test generating multiple report types from the same reporter instance."""

    def test_same_reporter_generates_all_types(self, reporter):
        sim = Simulator(ScenarioConfig(
            max_depth=2, max_replicas=5, cooldown_seconds=0.0, seed=42,
        )).run()
        threat = ThreatSimulator(ThreatConfig(
            max_depth=2, max_replicas=5, cooldown_seconds=0.0,
        )).run_all()
        comp = Comparator(ScenarioConfig(
            max_depth=2, max_replicas=5, cooldown_seconds=0.0, seed=42,
        )).compare_strategies(["greedy", "conservative"], seed=42)

        html_sim = reporter.simulation_report(sim)
        html_threat = reporter.threat_report(threat)
        html_comp = reporter.comparison_report(comp)
        html_combined = reporter.combined_report(
            simulation=sim, threat=threat, comparison=comp,
        )

        for html in [html_sim, html_threat, html_comp, html_combined]:
            assert "<!DOCTYPE html>" in html
            assert "</html>" in html

    def test_timestamp_consistent_across_reports(self, reporter):
        """All reports from the same instance should share the same timestamp."""
        sim = Simulator(ScenarioConfig(
            max_depth=1, max_replicas=3, cooldown_seconds=0.0, seed=1,
        )).run()
        html1 = reporter.simulation_report(sim)
        html2 = reporter.simulation_report(sim)
        # Both should have the same "Generated ..." timestamp
        assert reporter._timestamp in html1
        assert reporter._timestamp in html2


# ─── Special character handling ─────────────────────────────────────


class TestSpecialCharacters:
    """Ensure HTML-special characters are properly escaped in output."""

    def test_xss_in_worker_id(self, reporter):
        """Worker IDs with HTML tags should be escaped."""
        workers = {
            "<script>alert(1)</script>": WorkerRecord(
                worker_id="<script>alert(1)</script>",
                parent_id=None, depth=0,
                tasks_completed=1, children=[],
            ),
        }
        report = _make_sim_report(workers=workers)
        html = reporter.simulation_report(report)
        # The raw <script> should NOT appear unescaped
        assert "<script>alert(1)</script>" not in html
        assert "&lt;script&gt;" in html

    def test_xss_in_threat_name(self, reporter):
        results = [
            ThreatResult(
                "xss", '<img src=x onerror="alert(1)">',
                "Test <b>bold</b>",
                ThreatSeverity.HIGH, MitigationStatus.FAILED,
                attacks_attempted=1, attacks_blocked=0, attacks_succeeded=1,
                details=['<script>alert("xss")</script>'],
                duration_ms=0.1, audit_events=[],
            ),
        ]
        report = _make_threat_report(results)
        html = reporter.threat_report(report)
        assert '<img src=x onerror="alert(1)">' not in html
        assert '&lt;img' in html


# ─── Config table ───────────────────────────────────────────────────


class TestConfigTable:
    def test_config_table_contains_all_params(self, reporter):
        config = ScenarioConfig(
            max_depth=5, max_replicas=20, strategy="burst",
            tasks_per_worker=10, cooldown_seconds=3.0,
            cpu_limit=4, memory_limit_mb=512,
        )
        html = reporter._config_table(config)
        assert "burst" in html
        assert "5" in html
        assert "20" in html
        assert "3.0s" in html or "3s" in html
        assert "10" in html
        assert "4 cores" in html
        assert "512 MB" in html


# ─── Timeline icons ────────────────────────────────────────────────


class TestTimelineRendering:
    def test_timeline_icons_map_to_event_types(self, reporter):
        timeline = [
            {"type": "spawn", "time_ms": 0.0, "worker_id": "root", "detail": "Spawned"},
            {"type": "task", "time_ms": 1.0, "worker_id": "root", "detail": "Task done"},
            {"type": "replicate_ok", "time_ms": 2.0, "worker_id": "root", "detail": "Replicated"},
            {"type": "replicate_denied", "time_ms": 3.0, "worker_id": "root", "detail": "Denied"},
            {"type": "shutdown", "time_ms": 4.0, "worker_id": "root", "detail": "Shutdown"},
        ]
        html = reporter._render_timeline_html(timeline)
        assert "🟢" in html  # spawn
        assert "⚡" in html  # task
        assert "🔀" in html  # replicate_ok
        assert "🚫" in html  # replicate_denied
        assert "🔴" in html  # shutdown

    def test_timeline_unknown_type_uses_bullet(self, reporter):
        timeline = [
            {"type": "unknown_event", "time_ms": 0.0, "worker_id": "x", "detail": "Custom"},
        ]
        html = reporter._render_timeline_html(timeline)
        assert "•" in html

    def test_timeline_empty_list(self, reporter):
        html = reporter._render_timeline_html([])
        assert "timeline" in html
        assert "timeline-entry" not in html
