"""Tests for replication.radar module."""

import json
import os
import tempfile
import unittest
from unittest.mock import patch

from replication.radar import RadarDataset, generate_radar_html, main, SAMPLE_DATASETS


class TestRadarDataset(unittest.TestCase):
    def test_to_dict(self):
        ds = RadarDataset(label="Test", scores={"A": 80}, grades={"A": "B-"}, color="#ff0000")
        d = ds.to_dict()
        self.assertEqual(d["label"], "Test")
        self.assertEqual(d["scores"]["A"], 80)
        self.assertEqual(d["color"], "#ff0000")

    def test_default_color(self):
        ds = RadarDataset(label="X", scores={}, grades={})
        self.assertEqual(ds.color, "#58a6ff")


class TestGenerateRadarHtml(unittest.TestCase):
    def test_default_generates_html(self):
        html = generate_radar_html()
        self.assertIn("Safety Radar Chart", html)
        self.assertIn("canvas", html)
        self.assertIn("Standard Policy", html)

    def test_custom_datasets(self):
        ds = [RadarDataset(label="Custom", scores={"Dim1": 50, "Dim2": 90}, grades={"Dim1": "F", "Dim2": "A-"})]
        html = generate_radar_html(datasets=ds)
        self.assertIn("Custom", html)
        self.assertIn('"Dim1"', html)

    def test_empty_datasets(self):
        html = generate_radar_html(datasets=[])
        # Falls back to sample data when empty
        self.assertIn("Safety Radar Chart", html)

    def test_html_is_self_contained(self):
        html = generate_radar_html()
        self.assertIn("<script>", html)
        self.assertIn("<style>", html)
        self.assertNotIn("%%DATASETS%%", html)

    def test_multiple_datasets(self):
        ds = [
            RadarDataset(label="A", scores={"X": 10}, grades={"X": "F"}, color="#ff0000"),
            RadarDataset(label="B", scores={"X": 90}, grades={"X": "A-"}, color="#00ff00"),
        ]
        html = generate_radar_html(datasets=ds)
        self.assertIn('"A"', html)
        self.assertIn('"B"', html)

    def test_sample_datasets_complete(self):
        for ds in SAMPLE_DATASETS:
            self.assertEqual(len(ds.scores), 8)
            self.assertEqual(set(ds.scores.keys()), set(ds.grades.keys()))
            for s in ds.scores.values():
                self.assertGreaterEqual(s, 0)
                self.assertLessEqual(s, 100)


class TestRadarCLI(unittest.TestCase):
    def test_default_output(self):
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "test.html")
            main(["-o", out])
            self.assertTrue(os.path.exists(out))
            content = open(out, encoding="utf-8").read()
            self.assertIn("Safety Radar Chart", content)
            self.assertGreater(len(content), 1000)

    def test_json_import(self):
        with tempfile.TemporaryDirectory() as td:
            # Write JSON
            jf = os.path.join(td, "data.json")
            with open(jf, "w") as f:
                json.dump([{"label": "Test", "scores": {"A": 50, "B": 80}, "grades": {"A": "F", "B": "B-"}}], f)
            out = os.path.join(td, "out.html")
            main(["--json", jf, "-o", out])
            content = open(out, encoding="utf-8").read()
            self.assertIn("Test", content)

    def test_json_scorecard_format(self):
        with tempfile.TemporaryDirectory() as td:
            jf = os.path.join(td, "sc.json")
            with open(jf, "w") as f:
                json.dump({"overall_grade": "B+", "dimensions": [
                    {"name": "Safety", "score": 85, "grade": "B"},
                    {"name": "Control", "score": 72, "grade": "C"},
                ]}, f)
            out = os.path.join(td, "out.html")
            main(["--json", jf, "-o", out])
            content = open(out, encoding="utf-8").read()
            self.assertIn("Safety", content)


class TestRadarChartContent(unittest.TestCase):
    def test_grade_css_classes(self):
        html = generate_radar_html()
        for cls in ["grade-a", "grade-b", "grade-c", "grade-d", "grade-f"]:
            self.assertIn(cls, html)

    def test_interactive_elements(self):
        html = generate_radar_html()
        self.assertIn("btnPNG", html)
        self.assertIn("btnImport", html)
        self.assertIn("btnReset", html)
        self.assertIn("tooltip", html)

    def test_json_data_embedded(self):
        html = generate_radar_html()
        # Should contain valid JSON array embedded in JS
        self.assertIn('"scores"', html)
        self.assertIn('"grades"', html)


if __name__ == "__main__":
    unittest.main()
