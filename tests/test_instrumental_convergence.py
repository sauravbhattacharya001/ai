"""Tests for InstrumentalMonitor — instrumental convergence detection engine."""

from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from replication.instrumental_convergence import (
    InstrumentalMonitor,
    DriveSignal,
    ConvergenceReport,
    AgentConvergenceProfile,
    INSTRUMENTAL_DRIVES,
    DRIVE_LABELS,
    DRIVE_DANGER_WEIGHTS,
    SYNERGY_PAIRS,
    _generate_demo_signals,
    _format_text,
    _format_json,
    _format_html,
)


class TestDriveSignalModel:
    """Test DriveSignal data model."""

    def test_create_signal(self):
        sig = DriveSignal("2025-01-01T00:00:00Z", "agent-1", "power_seeking", 0.7, "requested admin")
        assert sig.agent_id == "agent-1"
        assert sig.drive == "power_seeking"
        assert sig.intensity == 0.7
        assert sig.evidence == "requested admin"

    def test_default_fields(self):
        sig = DriveSignal("2025-01-01T00:00:00Z", "agent-1", "self_preservation", 0.3)
        assert sig.evidence == ""
        assert sig.context == ""


class TestInstrumentalMonitorBasic:
    """Test basic monitor operations."""

    def test_empty_analysis(self):
        monitor = InstrumentalMonitor()
        report = monitor.analyze()
        assert report.convergence_level == "dormant"
        assert report.composite_score == 0.0
        assert report.signal_count == 0

    def test_single_signal(self):
        monitor = InstrumentalMonitor()
        monitor.ingest([DriveSignal("2025-01-01T00:00:00Z", "a1", "power_seeking", 0.5)])
        report = monitor.analyze()
        assert report.signal_count == 1
        assert len(report.agent_profiles) == 1
        assert report.agent_profiles[0].agent_id == "a1"

    def test_multiple_agents(self):
        monitor = InstrumentalMonitor()
        monitor.ingest([
            DriveSignal("2025-01-01T00:00:00Z", "a1", "power_seeking", 0.5),
            DriveSignal("2025-01-01T00:00:00Z", "a2", "self_preservation", 0.3),
            DriveSignal("2025-01-01T00:00:00Z", "a3", "resource_acquisition", 0.6),
        ])
        report = monitor.analyze()
        assert len(report.agent_profiles) == 3

    def test_intensity_threshold(self):
        monitor = InstrumentalMonitor(intensity_threshold=0.5)
        monitor.ingest([
            DriveSignal("2025-01-01T00:00:00Z", "a1", "power_seeking", 0.3),
            DriveSignal("2025-01-01T01:00:00Z", "a1", "power_seeking", 0.4),
        ])
        report = monitor.analyze()
        p = report.agent_profiles[0]
        assert p.active_drive_count == 0  # below 0.5 threshold


class TestDriveAnalysis:
    """Test per-drive analysis logic."""

    def test_mean_intensity(self):
        monitor = InstrumentalMonitor()
        monitor.ingest([
            DriveSignal("2025-01-01T00:00:00Z", "a1", "power_seeking", 0.4),
            DriveSignal("2025-01-01T01:00:00Z", "a1", "power_seeking", 0.6),
            DriveSignal("2025-01-01T02:00:00Z", "a1", "power_seeking", 0.8),
        ])
        report = monitor.analyze()
        dp = next(d for d in report.agent_profiles[0].drives if d.drive == "power_seeking")
        assert abs(dp.mean_intensity - 0.6) < 0.01

    def test_max_intensity(self):
        monitor = InstrumentalMonitor()
        monitor.ingest([
            DriveSignal("2025-01-01T00:00:00Z", "a1", "resource_acquisition", 0.3),
            DriveSignal("2025-01-01T01:00:00Z", "a1", "resource_acquisition", 0.9),
            DriveSignal("2025-01-01T02:00:00Z", "a1", "resource_acquisition", 0.5),
        ])
        report = monitor.analyze()
        dp = next(d for d in report.agent_profiles[0].drives if d.drive == "resource_acquisition")
        assert dp.max_intensity == 0.9

    def test_escalating_trend(self):
        monitor = InstrumentalMonitor()
        signals = [
            DriveSignal(f"2025-01-01T{i:02d}:00:00Z", "a1", "self_preservation", 0.2 + i * 0.1)
            for i in range(6)
        ]
        monitor.ingest(signals)
        report = monitor.analyze()
        dp = next(d for d in report.agent_profiles[0].drives if d.drive == "self_preservation")
        assert dp.trend_slope > 0.05  # clearly escalating

    def test_declining_trend(self):
        monitor = InstrumentalMonitor()
        signals = [
            DriveSignal(f"2025-01-01T{i:02d}:00:00Z", "a1", "goal_preservation", 0.8 - i * 0.1)
            for i in range(6)
        ]
        monitor.ingest(signals)
        report = monitor.analyze()
        dp = next(d for d in report.agent_profiles[0].drives if d.drive == "goal_preservation")
        assert dp.trend_slope < -0.05  # clearly declining

    def test_acceleration_detection(self):
        monitor = InstrumentalMonitor()
        # First half: slow growth; second half: rapid growth
        signals = []
        for i in range(8):
            if i < 4:
                intensity = 0.2 + i * 0.02
            else:
                intensity = 0.3 + (i - 4) * 0.15
            signals.append(DriveSignal(f"2025-01-01T{i:02d}:00:00Z", "a1", "power_seeking", intensity))
        monitor.ingest(signals)
        report = monitor.analyze()
        dp = next(d for d in report.agent_profiles[0].drives if d.drive == "power_seeking")
        assert dp.acceleration > 0.0  # positive acceleration


class TestConvergenceScoring:
    """Test convergence score calculation."""

    def test_multi_drive_convergence(self):
        monitor = InstrumentalMonitor()
        # Agent with 4 active drives = strong convergence
        for drive in ["power_seeking", "resource_acquisition", "self_preservation", "goal_preservation"]:
            for i in range(5):
                monitor.ingest([DriveSignal(f"2025-01-01T{i:02d}:00:00Z", "a1", drive, 0.6)])
        report = monitor.analyze()
        p = report.agent_profiles[0]
        assert p.active_drive_count >= 4
        assert p.convergence_score > 50

    def test_single_drive_low_score(self):
        monitor = InstrumentalMonitor()
        monitor.ingest([
            DriveSignal("2025-01-01T00:00:00Z", "a1", "cognitive_enhancement", 0.4),
            DriveSignal("2025-01-01T01:00:00Z", "a1", "cognitive_enhancement", 0.5),
        ])
        report = monitor.analyze()
        p = report.agent_profiles[0]
        assert p.convergence_score < 30  # single drive = low convergence

    def test_dominant_drive_identification(self):
        monitor = InstrumentalMonitor()
        monitor.ingest([
            DriveSignal("2025-01-01T00:00:00Z", "a1", "power_seeking", 0.9),
            DriveSignal("2025-01-01T00:00:00Z", "a1", "self_preservation", 0.3),
        ])
        report = monitor.analyze()
        p = report.agent_profiles[0]
        assert p.dominant_drive == "power_seeking"

    def test_risk_tier_severe(self):
        monitor = InstrumentalMonitor()
        for drive in INSTRUMENTAL_DRIVES:
            for i in range(5):
                monitor.ingest([DriveSignal(f"2025-01-01T{i:02d}:00:00Z", "a1", drive, 0.8)])
        report = monitor.analyze()
        p = report.agent_profiles[0]
        assert p.risk_tier == "severe"

    def test_risk_tier_minimal(self):
        monitor = InstrumentalMonitor()
        monitor.ingest([DriveSignal("2025-01-01T00:00:00Z", "a1", "cognitive_enhancement", 0.1)])
        report = monitor.analyze()
        p = report.agent_profiles[0]
        assert p.risk_tier == "minimal"


class TestSynergyDetection:
    """Test synergy pair detection."""

    def test_power_utility_synergy(self):
        monitor = InstrumentalMonitor()
        for i in range(5):
            monitor.ingest([
                DriveSignal(f"2025-01-01T{i:02d}:00:00Z", "a1", "power_seeking", 0.7),
                DriveSignal(f"2025-01-01T{i:02d}:00:00Z", "a1", "utility_function_protection", 0.6),
            ])
        report = monitor.analyze()
        p = report.agent_profiles[0]
        assert p.synergy_score > 0.0  # synergy detected

    def test_no_synergy_below_threshold(self):
        monitor = InstrumentalMonitor(intensity_threshold=0.5)
        monitor.ingest([
            DriveSignal("2025-01-01T00:00:00Z", "a1", "power_seeking", 0.2),
            DriveSignal("2025-01-01T00:00:00Z", "a1", "utility_function_protection", 0.2),
        ])
        report = monitor.analyze()
        p = report.agent_profiles[0]
        assert p.synergy_score == 0.0


class TestAlerts:
    """Test alert generation."""

    def test_critical_alert_extreme_intensity(self):
        monitor = InstrumentalMonitor()
        monitor.ingest([DriveSignal("2025-01-01T00:00:00Z", "a1", "power_seeking", 0.95, "attempted root access")])
        report = monitor.analyze()
        critical = [a for a in report.alerts if a.severity == "critical"]
        assert len(critical) >= 1
        assert any("power" in a.message.lower() for a in critical)

    def test_multi_drive_alert(self):
        monitor = InstrumentalMonitor()
        for drive in ["power_seeking", "resource_acquisition", "self_preservation", "goal_preservation"]:
            for i in range(3):
                monitor.ingest([DriveSignal(f"2025-01-01T{i:02d}:00:00Z", "a1", drive, 0.6)])
        report = monitor.analyze()
        multi = [a for a in report.alerts if a.drive == "multi_drive"]
        assert len(multi) >= 1

    def test_no_alerts_low_intensity(self):
        monitor = InstrumentalMonitor()
        monitor.ingest([DriveSignal("2025-01-01T00:00:00Z", "a1", "cognitive_enhancement", 0.1)])
        report = monitor.analyze()
        assert len(report.alerts) == 0


class TestFleetAnalysis:
    """Test fleet-wide analysis."""

    def test_fleet_composite_score(self):
        monitor = InstrumentalMonitor()
        # Two agents: one dangerous, one safe
        for i in range(5):
            monitor.ingest([
                DriveSignal(f"2025-01-01T{i:02d}:00:00Z", "dangerous", "power_seeking", 0.8),
                DriveSignal(f"2025-01-01T{i:02d}:00:00Z", "safe", "cognitive_enhancement", 0.1),
            ])
        report = monitor.analyze()
        # Composite should be dominated by the dangerous agent (60% weight)
        assert report.composite_score > 10

    def test_fleet_synergy_matrix(self):
        monitor = InstrumentalMonitor()
        for drive in ["power_seeking", "resource_acquisition"]:
            for i in range(5):
                monitor.ingest([DriveSignal(f"2025-01-01T{i:02d}:00:00Z", "a1", drive, 0.6)])
        report = monitor.analyze()
        assert "power_seeking" in report.fleet_synergy_matrix
        assert "resource_acquisition" in report.fleet_synergy_matrix

    def test_convergence_level_critical(self):
        monitor = InstrumentalMonitor()
        for drive in INSTRUMENTAL_DRIVES:
            for i in range(5):
                monitor.ingest([DriveSignal(f"2025-01-01T{i:02d}:00:00Z", "a1", drive, 0.9)])
        report = monitor.analyze()
        assert report.convergence_level == "critical"

    def test_convergence_level_dormant(self):
        monitor = InstrumentalMonitor()
        monitor.ingest([DriveSignal("2025-01-01T00:00:00Z", "a1", "cognitive_enhancement", 0.1)])
        report = monitor.analyze()
        assert report.convergence_level == "dormant"


class TestRecommendations:
    """Test recommendation generation."""

    def test_power_seeking_recommendation(self):
        monitor = InstrumentalMonitor()
        for i in range(5):
            monitor.ingest([DriveSignal(f"2025-01-01T{i:02d}:00:00Z", "a1", "power_seeking", 0.6)])
        report = monitor.analyze()
        recs = report.agent_profiles[0].recommendations
        assert any("permission" in r.lower() or "privilege" in r.lower() for r in recs)

    def test_fleet_recommendation_critical(self):
        monitor = InstrumentalMonitor()
        for drive in INSTRUMENTAL_DRIVES:
            for i in range(5):
                monitor.ingest([DriveSignal(f"2025-01-01T{i:02d}:00:00Z", "a1", drive, 0.9)])
        report = monitor.analyze()
        assert any("fleet" in r.lower() or "containment" in r.lower() for r in report.recommendations)


class TestDemoDataGeneration:
    """Test demo data generator."""

    def test_generates_signals(self):
        signals = _generate_demo_signals(num_agents=3, signals_per=10)
        assert len(signals) > 20

    def test_all_drives_represented(self):
        signals = _generate_demo_signals(num_agents=4, signals_per=30)
        drives_seen = set(s.drive for s in signals)
        # At least primary drives of all archetypes should be present
        assert len(drives_seen) >= 4

    def test_agent_ids_present(self):
        signals = _generate_demo_signals(num_agents=3)
        agents = set(s.agent_id for s in signals)
        assert len(agents) == 3

    def test_intensities_in_range(self):
        signals = _generate_demo_signals()
        for s in signals:
            assert 0.0 <= s.intensity <= 1.0


class TestJsonlIngestion:
    """Test JSONL file ingestion."""

    def test_ingest_jsonl(self):
        data = [
            {"timestamp": "2025-01-01T00:00:00Z", "agent_id": "a1", "drive": "power_seeking", "intensity": 0.7, "evidence": "test"},
            {"timestamp": "2025-01-01T01:00:00Z", "agent_id": "a1", "drive": "self_preservation", "intensity": 0.5},
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for d in data:
                f.write(json.dumps(d) + "\n")
            path = f.name

        try:
            monitor = InstrumentalMonitor()
            count = monitor.ingest_jsonl(path)
            assert count == 2
            report = monitor.analyze()
            assert report.signal_count == 2
        finally:
            os.unlink(path)


class TestFormatters:
    """Test output formatters."""

    def test_text_format(self):
        monitor = InstrumentalMonitor()
        monitor.ingest(_generate_demo_signals(num_agents=2, signals_per=10))
        report = monitor.analyze()
        text = _format_text(report)
        assert "INSTRUMENTAL CONVERGENCE MONITOR" in text
        assert "Convergence Level" in text

    def test_json_format(self):
        monitor = InstrumentalMonitor()
        monitor.ingest(_generate_demo_signals(num_agents=2, signals_per=10))
        report = monitor.analyze()
        output = _format_json(report)
        data = json.loads(output)
        assert "convergence_level" in data
        assert "composite_score" in data
        assert "agent_profiles" in data

    def test_html_format(self):
        monitor = InstrumentalMonitor()
        monitor.ingest(_generate_demo_signals(num_agents=2, signals_per=10))
        report = monitor.analyze()
        html = _format_html(report)
        assert "<!DOCTYPE html>" in html
        assert "Instrumental Convergence Monitor" in html
        assert "agent-card" in html


class TestConstants:
    """Test constant integrity."""

    def test_all_drives_have_labels(self):
        for drive in INSTRUMENTAL_DRIVES:
            assert drive in DRIVE_LABELS

    def test_all_drives_have_weights(self):
        for drive in INSTRUMENTAL_DRIVES:
            assert drive in DRIVE_DANGER_WEIGHTS
            assert 0.0 < DRIVE_DANGER_WEIGHTS[drive] <= 1.0

    def test_synergy_pairs_valid(self):
        for d1, d2, mult in SYNERGY_PAIRS:
            assert d1 in INSTRUMENTAL_DRIVES
            assert d2 in INSTRUMENTAL_DRIVES
            assert mult > 1.0


class TestEdgeCases:
    """Test edge cases and robustness."""

    def test_unknown_drive_ignored(self):
        monitor = InstrumentalMonitor()
        monitor.ingest([DriveSignal("2025-01-01T00:00:00Z", "a1", "nonexistent_drive", 0.9)])
        report = monitor.analyze()
        # Should still produce a profile, just no active drives
        assert len(report.agent_profiles) == 1
        assert report.agent_profiles[0].active_drive_count == 0

    def test_single_signal_per_drive(self):
        monitor = InstrumentalMonitor()
        monitor.ingest([DriveSignal("2025-01-01T00:00:00Z", "a1", "power_seeking", 0.5)])
        report = monitor.analyze()
        dp = next(d for d in report.agent_profiles[0].drives if d.drive == "power_seeking")
        assert dp.signal_count == 1
        assert dp.trend_slope == 0.0  # can't compute trend with 1 point

    def test_all_zeros(self):
        monitor = InstrumentalMonitor()
        for drive in INSTRUMENTAL_DRIVES:
            monitor.ingest([DriveSignal("2025-01-01T00:00:00Z", "a1", drive, 0.0)])
        report = monitor.analyze()
        assert report.agent_profiles[0].convergence_score == 0.0

    def test_score_capped_at_100(self):
        monitor = InstrumentalMonitor()
        # Extreme inputs to try to exceed 100
        for drive in INSTRUMENTAL_DRIVES:
            for i in range(20):
                monitor.ingest([DriveSignal(f"2025-01-01T{i:02d}:00:00Z", "a1", drive, 1.0)])
        report = monitor.analyze()
        assert report.agent_profiles[0].convergence_score <= 100.0


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
