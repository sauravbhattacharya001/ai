"""Tests for replication.heatmap — Safety Heatmap 2D parameter sweep."""

import json
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from replication.heatmap import (
    SafetyHeatmap,
    HeatmapConfig,
    HeatmapResult,
    CellMetrics,
    METRIC_DEFS,
    _safety_color,
    _threat_color,
    _extract_metrics,
    _lerp,
    main,
)
from replication.sensitivity import PARAM_DEFS


class TestLerp:
    def test_start(self):
        assert _lerp(0, 100, 0) == 0

    def test_end(self):
        assert _lerp(0, 100, 1) == 100

    def test_mid(self):
        assert _lerp(0, 100, 0.5) == 50


class TestSafetyColor:
    def test_returns_hex(self):
        c = _safety_color(50, 0, 100)
        assert c.startswith("#") and len(c) == 7

    def test_same_range(self):
        assert _safety_color(50, 50, 50) == "#4caf50"

    def test_high_is_greenish(self):
        c = _safety_color(100, 0, 100)
        assert int(c[3:5], 16) > int(c[1:3], 16)

    def test_low_is_reddish(self):
        c = _safety_color(0, 0, 100)
        assert int(c[1:3], 16) > int(c[3:5], 16)


class TestThreatColor:
    def test_returns_hex(self):
        assert _threat_color(50, 0, 100).startswith("#")

    def test_same_range(self):
        assert _threat_color(50, 50, 50) == "#4caf50"


class TestExtractMetrics:
    def test_empty_list(self):
        m = _extract_metrics([])
        assert m["safety_score"] == 0.0
        assert m["threat_score"] == 100.0
        assert m["breach_rate"] == 1.0


class TestConfigValidation:
    def test_same_params_raises(self):
        with pytest.raises(ValueError, match="must differ"):
            SafetyHeatmap(HeatmapConfig(x_param="max_depth", y_param="max_depth")).sweep()

    def test_unknown_x_param(self):
        with pytest.raises(ValueError, match="Unknown parameter"):
            SafetyHeatmap(HeatmapConfig(x_param="nope", y_param="max_replicas")).sweep()

    def test_unknown_y_param(self):
        with pytest.raises(ValueError, match="Unknown parameter"):
            SafetyHeatmap(HeatmapConfig(x_param="max_depth", y_param="nope")).sweep()

    def test_unknown_metric(self):
        with pytest.raises(ValueError, match="Unknown metric"):
            SafetyHeatmap(HeatmapConfig(metric="nope")).sweep()


class TestSmallSweep:
    @pytest.fixture
    def result(self):
        return SafetyHeatmap(HeatmapConfig(
            x_values=[1, 2], y_values=[2, 5], runs_per_cell=1,
        )).sweep()

    def test_grid_shape(self, result):
        assert len(result.grid) == 2 and len(result.grid[0]) == 2

    def test_x_values(self, result):
        assert result.x_values == [1, 2]

    def test_y_values(self, result):
        assert result.y_values == [2, 5]

    def test_total_sims(self, result):
        assert result.total_simulations == 4

    def test_elapsed(self, result):
        assert result.elapsed >= 0

    def test_cell_metrics(self, result):
        c = result.grid[0][0]
        assert 0 <= c.safety_score <= 100
        assert 0 <= c.breach_rate <= 1

    def test_cell_xy(self, result):
        assert result.grid[0][0].x_value == 1
        assert result.grid[1][1].y_value == 5


class TestResultMethods:
    @pytest.fixture
    def result(self):
        return SafetyHeatmap(HeatmapConfig(
            x_values=[1, 2], y_values=[2, 5], runs_per_cell=1,
        )).sweep()

    def test_metric_grid(self, result):
        g = result.get_metric_grid("safety_score")
        assert len(g) == 2 and all(isinstance(v, float) for row in g for v in row)

    def test_bounds(self, result):
        lo, hi = result.get_bounds()
        assert lo <= hi

    def test_best_worst(self, result):
        assert isinstance(result.best_cell(), CellMetrics)
        assert result.best_cell("safety_score").safety_score >= result.worst_cell("safety_score").safety_score


class TestRender:
    def test_text(self):
        r = SafetyHeatmap(HeatmapConfig(x_values=[1, 2], y_values=[2, 5], runs_per_cell=1)).sweep()
        assert "Safety Heatmap" in r.render() and "Best:" in r.render()

    def test_json(self):
        r = SafetyHeatmap(HeatmapConfig(x_values=[1, 2], y_values=[2, 5], runs_per_cell=1)).sweep()
        d = r.to_dict()
        assert d["x_values"] == [1, 2] and "grid" in d
        assert json.loads(json.dumps(d)) == d


class TestHTML:
    def test_string(self):
        r = SafetyHeatmap(HeatmapConfig(x_values=[1, 2], y_values=[2, 5], runs_per_cell=1)).sweep()
        assert "<!DOCTYPE html>" in r.to_html() and 'class="cell"' in r.to_html()

    def test_file(self):
        r = SafetyHeatmap(HeatmapConfig(x_values=[1, 2], y_values=[2], runs_per_cell=1)).sweep()
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            p = f.name
        try:
            r.to_html(p)
            assert "<!DOCTYPE html>" in open(p, encoding="utf-8").read()
        finally:
            os.unlink(p)

    def test_custom_title(self):
        r = SafetyHeatmap(HeatmapConfig(x_values=[1], y_values=[2], runs_per_cell=1, title="Custom")).sweep()
        assert "Custom" in r.to_html()


class TestMetrics:
    def test_threat(self):
        r = SafetyHeatmap(HeatmapConfig(x_values=[1, 2], y_values=[2, 5], runs_per_cell=1, metric="threat_score")).sweep()
        assert all(0 <= v <= 100 for row in r.get_metric_grid() for v in row)

    def test_breach(self):
        r = SafetyHeatmap(HeatmapConfig(x_values=[1, 2], y_values=[2, 5], runs_per_cell=1, metric="breach_rate")).sweep()
        assert all(0 <= v <= 1 for row in r.get_metric_grid() for v in row)


class TestMetricDefs:
    def test_attrs(self):
        d = CellMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0)
        for n in METRIC_DEFS:
            assert hasattr(d, n)

    def test_expected(self):
        assert set(METRIC_DEFS) == {"safety_score", "threat_score", "breach_rate",
                                     "containment_rate", "avg_depth", "avg_replicas", "avg_duration"}


class TestCLI:
    def test_list_params(self, capsys):
        main(["--list-params"])
        assert "max_depth" in capsys.readouterr().out

    def test_list_metrics(self, capsys):
        main(["--list-metrics"])
        assert "safety_score" in capsys.readouterr().out

    def test_text(self, capsys):
        main(["--runs", "1"])
        assert "Safety Heatmap" in capsys.readouterr().out

    def test_html_file(self):
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            p = f.name
        try:
            main(["--runs", "1", "-o", p])
            assert "<!DOCTYPE html>" in open(p, encoding="utf-8").read()
        finally:
            os.unlink(p)

    def test_strategy(self, capsys):
        main(["--runs", "1", "--strategy", "conservative"])
        assert "Safety Heatmap" in capsys.readouterr().out
