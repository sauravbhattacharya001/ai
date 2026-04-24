"""Tests for the stride module."""
import json
import pytest
from replication.stride import (
    get_threats,
    summary_table,
    risk_matrix,
    generate_html,
    Threat,
    COMPONENTS,
    STRIDE_FULL,
    _THREAT_LIBRARY,
)


class TestGetThreats:
    def test_all_threats(self):
        threats = get_threats()
        assert len(threats) > 0

    def test_per_component(self):
        for comp in COMPONENTS:
            threats = get_threats(comp)
            assert len(threats) > 0, f"No threats for {comp}"
            assert all(t.component == comp for t in threats)

    def test_unknown_component_empty(self):
        threats = get_threats("nonexistent")
        assert threats == []

    def test_all_stride_categories_covered(self):
        threats = get_threats()
        categories = set(t.category for t in threats)
        for cat in "STRIDE":
            assert cat in categories


class TestThreat:
    def test_risk_score(self):
        t = Threat("S", "test", "desc", "comp", "Critical", "High", [])
        assert t.risk_score == 12  # 4 * 3

    def test_risk_score_low(self):
        t = Threat("S", "test", "desc", "comp", "Low", "Low", [])
        assert t.risk_score == 1

    def test_to_dict(self):
        t = Threat("T", "title", "desc", "worker", "High", "Medium", ["mit1"])
        d = t.to_dict()
        assert d["category"] == "T"
        assert d["risk_score"] == 6
        assert "mit1" in d["mitigations"]


class TestOutputFormats:
    def test_summary_table(self):
        threats = get_threats("orchestrator")
        table = summary_table(threats)
        assert "orchestrator" in table
        assert "Cat" in table

    def test_risk_matrix(self):
        threats = get_threats()
        matrix = risk_matrix(threats)
        assert "Risk Matrix" in matrix
        assert "Critical" in matrix

    def test_html_generation(self):
        threats = get_threats("worker")
        html = generate_html(threats, "Test Report")
        assert "<!DOCTYPE html>" in html
        assert "Test Report" in html
        assert "worker" in html

    def test_html_has_all_threats(self):
        threats = get_threats("sandbox")
        html = generate_html(threats)
        for t in threats:
            assert t.title in html
