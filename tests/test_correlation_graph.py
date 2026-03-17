"""Tests for correlation_graph module."""

from __future__ import annotations

import unittest
from pathlib import Path

from replication.correlation_graph import generate_correlation_graph, main


class TestGenerateCorrelationGraph(unittest.TestCase):
    """Tests for the HTML generation function."""

    def test_returns_string(self):
        html = generate_correlation_graph()
        self.assertIsInstance(html, str)

    def test_contains_doctype(self):
        html = generate_correlation_graph()
        self.assertIn("<!DOCTYPE html>", html)

    def test_contains_title(self):
        html = generate_correlation_graph()
        self.assertIn("Threat Correlation Graph", html)

    def test_contains_canvas(self):
        html = generate_correlation_graph()
        self.assertIn('<canvas id="graph"', html)

    def test_contains_source_filters(self):
        html = generate_correlation_graph()
        for src in ["drift", "escalation", "killchain", "canary", "covert", "behavior"]:
            self.assertIn(f'data-src="{src}"', html)

    def test_contains_severity_filters(self):
        html = generate_correlation_graph()
        for sev in ["critical", "high", "medium", "low"]:
            self.assertIn(f'data-sev="{sev}"', html)

    def test_contains_agent_filter(self):
        html = generate_correlation_graph()
        self.assertIn('id="agentFilter"', html)

    def test_contains_timeline_slider(self):
        html = generate_correlation_graph()
        self.assertIn('id="timeSlider"', html)

    def test_contains_export_button(self):
        html = generate_correlation_graph()
        self.assertIn('id="exportBtn"', html)

    def test_contains_stats_panel(self):
        html = generate_correlation_graph()
        self.assertIn('id="statsPanel"', html)

    def test_contains_sidebar(self):
        html = generate_correlation_graph()
        self.assertIn('id="sidebar"', html)

    def test_contains_legend(self):
        html = generate_correlation_graph()
        self.assertIn("legend", html.lower())

    def test_contains_force_simulation(self):
        html = generate_correlation_graph()
        self.assertIn("simulate", html)

    def test_contains_demo_data_generation(self):
        html = generate_correlation_graph()
        self.assertIn("generateData", html)

    def test_contains_severity_colors(self):
        html = generate_correlation_graph()
        self.assertIn("#f06060", html)  # critical
        self.assertIn("#4ade80", html)  # info

    def test_html_is_self_contained(self):
        html = generate_correlation_graph()
        # No external script/css references
        self.assertNotIn("src=\"http", html)
        self.assertNotIn("href=\"http", html)

    def test_consistent_output(self):
        a = generate_correlation_graph()
        b = generate_correlation_graph()
        self.assertEqual(a, b)


class TestMain(unittest.TestCase):
    """Tests for CLI entry point."""

    def test_writes_file(self):
        import tempfile, os
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "test_graph.html")
            main(["-o", out])
            self.assertTrue(os.path.exists(out))
            content = Path(out).read_text(encoding="utf-8")
            self.assertIn("<!DOCTYPE html>", content)

    def test_default_filename(self):
        import tempfile, os
        orig = os.getcwd()
        with tempfile.TemporaryDirectory() as td:
            os.chdir(td)
            try:
                main([])
                self.assertTrue(os.path.exists("correlation_graph.html"))
            finally:
                os.chdir(orig)

    def test_custom_output_path(self):
        import tempfile, os
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "custom.html")
            main(["-o", out])
            self.assertTrue(os.path.exists(out))
            self.assertGreater(os.path.getsize(out), 1000)


if __name__ == "__main__":
    unittest.main()
