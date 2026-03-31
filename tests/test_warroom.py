"""Tests for the warroom module."""

from __future__ import annotations

from replication.warroom import generate_warroom, main


class TestGenerateWarroom:
    def test_returns_html(self):
        html = generate_warroom()
        assert "<!DOCTYPE html>" in html
        assert "War Room" in html

    def test_contains_fleet_grid(self):
        html = generate_warroom()
        assert "fleetGrid" in html

    def test_contains_kill_switch(self):
        html = generate_warroom()
        assert "killBtn" in html
        assert "toggleKill" in html

    def test_contains_event_feed(self):
        html = generate_warroom()
        assert "eventFeed" in html

    def test_contains_gauges(self):
        html = generate_warroom()
        assert "gauge-arc" in html

    def test_contains_timeline(self):
        html = generate_warroom()
        assert "timelineCanvas" in html

    def test_contains_severity_chart(self):
        html = generate_warroom()
        assert "sevChart" in html


class TestWarroomCLI:
    def test_main_writes_file(self, tmp_path):
        out = tmp_path / "warroom.html"
        main(["-o", str(out)])
        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert "War Room" in content
        assert len(content) > 1000
