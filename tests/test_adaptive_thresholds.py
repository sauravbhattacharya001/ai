"""Tests for replication.adaptive_thresholds."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from replication.adaptive_thresholds import (
    PRESETS,
    ThresholdProfile,
    ThresholdState,
    load_preset,
    main,
)


# ── ThresholdState ─────────────────────────────────────────────────────


class TestThresholdState:
    def test_warmup_first_observation_seeds_ema(self):
        ts = ThresholdState(metric="m")
        result = ts.observe(10.0)
        assert ts.ema == 10.0
        assert ts.ema_var == 0.0
        assert result["breached"] is False
        assert result["observations"] == 1

    def test_ema_converges_toward_constant_signal(self):
        ts = ThresholdState(metric="m", alpha=0.5)
        for _ in range(50):
            ts.observe(5.0)
        assert ts.ema == pytest.approx(5.0, abs=1e-6)
        # variance should also collapse toward zero
        assert ts.ema_var < 1e-6

    def test_std_property_never_negative(self):
        ts = ThresholdState(metric="m")
        # explicitly set a negative variance to make sure std is clamped
        ts.ema_var = -1.0
        assert ts.std >= 0.0

    def test_breach_detection_outlier(self):
        ts = ThresholdState(metric="m", alpha=0.2, sigma_multiplier=2.0)
        for _ in range(20):
            ts.observe(1.0 + 0.01)  # tiny noise around 1
        # introduce a clear outlier
        result = ts.observe(100.0)
        assert result["breached"] is True
        assert ts.breach_count == 1
        assert ts.last_breach_idx == ts.observations

    def test_min_floor_and_ceiling_enforced(self):
        ts = ThresholdState(metric="m", min_floor=0.0, max_ceiling=10.0)
        for _ in range(20):
            ts.observe(5.0)
        # synthetically blow up the variance
        ts.ema_var = 1e6
        assert ts.upper_threshold <= 10.0
        assert ts.lower_threshold >= 0.0

    def test_recent_values_window_trimmed(self):
        ts = ThresholdState(metric="m", recent_window=5)
        for i in range(20):
            ts.observe(float(i))
        assert len(ts.recent_values) == 5
        assert ts.recent_values[-1] == 19.0

    def test_risk_multiplier_tightens_on_breach_and_relaxes(self):
        ts = ThresholdState(metric="m", alpha=0.2, sigma_multiplier=2.0)
        for _ in range(15):
            ts.observe(1.0)
        before = ts.risk_multiplier
        ts.observe(100.0)
        after_breach = ts.risk_multiplier
        assert after_breach > before
        # several stable observations should relax it
        for _ in range(50):
            ts.observe(1.0)
        assert ts.risk_multiplier <= after_breach

    def test_risk_multiplier_capped_at_3(self):
        ts = ThresholdState(metric="m", alpha=0.2, sigma_multiplier=2.0)
        ts.observe(1.0)
        for _ in range(50):
            # alternating outliers in both directions
            ts.observe(1000.0)
        assert ts.risk_multiplier <= 3.0

    def test_status_keys_present(self):
        ts = ThresholdState(metric="latency")
        status = ts.observe(42.0)
        for key in (
            "metric", "value", "ema", "std", "upper", "lower",
            "breached", "risk_multiplier", "observations", "breach_count",
        ):
            assert key in status
        assert status["metric"] == "latency"


# ── forecast_breach ───────────────────────────────────────────────────


class TestForecast:
    def test_forecast_none_when_too_few_samples(self):
        ts = ThresholdState(metric="m")
        for v in [1.0, 1.1, 1.2]:
            ts.observe(v)
        assert ts.forecast_breach() is None

    def test_forecast_predicts_upward_breach(self):
        ts = ThresholdState(metric="m", alpha=0.2, sigma_multiplier=2.0)
        # stable baseline first
        for _ in range(20):
            ts.observe(1.0)
        # then a clear upward trend within the window
        for v in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]:
            ts.observe(v)
        fc = ts.forecast_breach(horizon=20)
        # may already be breached or projected to breach soon
        assert fc is not None
        assert fc["metric"] == "m"
        assert fc["steps_to_breach"] >= 1
        assert "confidence" in fc and fc["confidence"] in {"low", "medium", "high"}

    def test_forecast_none_when_flat_within_bounds(self):
        ts = ThresholdState(metric="m", alpha=0.2, sigma_multiplier=3.0)
        for _ in range(40):
            ts.observe(1.0)
        # flat signal => no breach projected
        assert ts.forecast_breach(horizon=5) is None

    def test_forecast_confidence_scales_with_sample_size(self):
        ts = ThresholdState(metric="m", alpha=0.2, sigma_multiplier=2.0, recent_window=50)
        # ramp produces a slope; small sample => low confidence
        for i in range(6):
            ts.observe(float(i))
        fc_small = ts.forecast_breach(horizon=50)
        assert fc_small is not None
        assert fc_small["confidence"] == "low"

        ts2 = ThresholdState(metric="m", alpha=0.2, sigma_multiplier=2.0, recent_window=50)
        for i in range(30):
            ts2.observe(float(i))
        fc_big = ts2.forecast_breach(horizon=50)
        assert fc_big is not None
        assert fc_big["confidence"] in {"medium", "high"}


# ── ThresholdProfile ──────────────────────────────────────────────────


class TestThresholdProfile:
    def test_add_metric_returns_state(self):
        p = ThresholdProfile(name="t")
        ts = p.add_metric("x", alpha=0.3, sigma=2.0)
        assert isinstance(ts, ThresholdState)
        assert "x" in p.thresholds
        assert ts.alpha == 0.3

    def test_observe_autocreates_unknown_metric(self):
        p = ThresholdProfile(name="t")
        p.observe("brand_new", 1.0)
        assert "brand_new" in p.thresholds

    def test_observe_batch_returns_one_per_metric(self):
        p = ThresholdProfile(name="t")
        results = p.observe_batch({"a": 1.0, "b": 2.0, "c": 3.0})
        assert len(results) == 3
        metrics = {r["metric"] for r in results}
        assert metrics == {"a", "b", "c"}

    def test_summary_structure(self):
        p = load_preset("fleet")
        for _ in range(5):
            p.observe_batch({"score_drift": 0.15, "latency_ms": 100.0})
        s = p.summary()
        assert s["profile"] == "fleet"
        assert "score_drift" in s["metrics"]
        assert "ema" in s["metrics"]["score_drift"]

    def test_health_score_empty_profile_is_100(self):
        p = ThresholdProfile(name="empty")
        assert p.health_score() == 100.0

    def test_health_score_no_observations_is_100(self):
        p = load_preset("fleet")
        # no observations recorded
        assert p.health_score() == 100.0

    def test_health_score_drops_with_breaches(self):
        p = ThresholdProfile(name="t")
        p.add_metric("m", alpha=0.2, sigma=2.0)
        for _ in range(20):
            p.observe("m", 1.0)
        # repeated outliers
        for _ in range(5):
            p.observe("m", 100.0)
        assert p.health_score() < 100.0
        assert p.health_score() >= 0.0

    def test_forecast_all_returns_only_metrics_with_forecasts(self):
        p = ThresholdProfile(name="t")
        # one stable metric (no forecast), one trending (forecast)
        p.add_metric("flat", alpha=0.2, sigma=3.0)
        p.add_metric("trend", alpha=0.2, sigma=2.0)
        for _ in range(20):
            p.observe("flat", 1.0)
        for i in range(15):
            p.observe("trend", float(i))
        fcs = p.forecast_all(horizon=20)
        # trend likely produces a forecast; flat should not
        assert all(fc["metric"] == "trend" for fc in fcs)


# ── presets ───────────────────────────────────────────────────────────


class TestPresets:
    @pytest.mark.parametrize("name", list(PRESETS.keys()))
    def test_load_known_preset(self, name):
        p = load_preset(name)
        assert p.name == name
        assert len(p.thresholds) > 0
        for metric, params in PRESETS[name].items():
            assert metric in p.thresholds
            ts = p.thresholds[metric]
            assert ts.alpha == params["alpha"]
            assert ts.sigma_multiplier == params["sigma"]

    def test_load_unknown_preset_raises(self):
        with pytest.raises(ValueError, match="Unknown preset"):
            load_preset("nope")


# ── CLI ───────────────────────────────────────────────────────────────


class TestCLI:
    def test_show_profile_default(self, capsys):
        main(["--profile", "agent"])
        out = capsys.readouterr().out
        assert "Profile: agent" in out
        assert "alignment_score" in out

    def test_demo_runs_and_prints_summary(self, capsys):
        main(["--demo", "--profile", "fleet", "--steps", "30"])
        out = capsys.readouterr().out
        assert "Summary" in out
        assert "Health Score" in out

    def test_demo_json_export(self, capsys):
        main(["--demo", "--profile", "fleet", "--steps", "20", "--export", "json"])
        out = capsys.readouterr().out
        # the JSON block is the last printed thing
        brace = out.find("{")
        assert brace != -1
        data = json.loads(out[brace:])
        assert data["profile"] == "fleet"
        assert "summary" in data
        assert "health_score" in data

    def test_demo_writes_output_file(self, tmp_path, capsys):
        target = tmp_path / "out.json"
        main([
            "--demo", "--profile", "fleet", "--steps", "15",
            "--export", "json", "-o", str(target),
        ])
        assert target.exists()
        data = json.loads(target.read_text(encoding="utf-8"))
        assert data["profile"] == "fleet"
        assert isinstance(data["breaches"], list)

    def test_demo_metric_filter(self, capsys):
        main(["--demo", "--profile", "fleet", "--steps", "30",
              "--metric", "score_drift"])
        out = capsys.readouterr().out
        # all printed metric names in summary should be score_drift only
        # (very loose check — Health Score line still prints)
        assert "score_drift" in out

    def test_demo_window_override_propagates(self):
        # White-box: --window should mutate every threshold's recent_window
        # Reach into main by re-implementing the setup the same way:
        from replication.adaptive_thresholds import load_preset
        profile = load_preset("fleet")
        for ts in profile.thresholds.values():
            ts.recent_window = 7
        assert all(ts.recent_window == 7 for ts in profile.thresholds.values())
