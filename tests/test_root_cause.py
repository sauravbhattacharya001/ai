"""Tests for replication.root_cause."""

from __future__ import annotations

import json

from replication.root_cause import (
    SEVERITY_LEVELS,
    analyze_fault_tree,
    analyze_fishbone,
    analyze_five_whys,
    format_html,
    format_markdown,
    format_text,
    full_analysis,
)


class TestFiveWhys:
    def test_returns_chain(self):
        result = analyze_five_whys("test incident", "medium")
        assert result["method"] == "5 Whys"
        assert len(result["chain"]) == 5
        assert "root_cause" in result

    def test_all_severities(self):
        for sev in SEVERITY_LEVELS:
            result = analyze_five_whys("test", sev)
            assert result["chain"]


class TestFishbone:
    def test_six_categories(self):
        result = analyze_fishbone("test", "high")
        assert len(result["categories"]) == 6
        for cat in ("Policy", "Monitoring", "Architecture", "Human", "Environment", "Data"):
            assert cat in result["categories"]

    def test_causes_non_empty(self):
        for sev in SEVERITY_LEVELS:
            result = analyze_fishbone("test", sev)
            for causes in result["categories"].values():
                assert len(causes) >= 1


class TestFaultTree:
    def test_has_cut_sets(self):
        result = analyze_fault_tree("test", "critical")
        assert result["minimal_cut_sets"]
        assert result["cut_set_count"] > 0

    def test_tree_structure(self):
        result = analyze_fault_tree("test", "high")
        tree = result["tree"]
        assert "top_event" in tree
        assert "gate" in tree
        assert "children" in tree


class TestFullAnalysis:
    def test_all_methods(self):
        result = full_analysis("escape", "critical", ["all"])
        assert len(result["analyses"]) == 3
        methods = {a["method"] for a in result["analyses"]}
        assert "5 Whys" in methods
        assert "Fishbone (Ishikawa)" in methods
        assert "Fault Tree Analysis" in methods

    def test_single_method(self):
        result = full_analysis("test", "low", ["5whys"])
        assert len(result["analyses"]) == 1

    def test_has_remediation(self):
        result = full_analysis("test", "high")
        assert len(result["remediation"]) > 0


class TestFormatters:
    def test_text(self):
        result = full_analysis("test", "medium")
        text = format_text(result)
        assert "ROOT CAUSE ANALYSIS" in text

    def test_markdown(self):
        result = full_analysis("test", "high")
        md = format_markdown(result)
        assert "# Root Cause Analysis" in md

    def test_html(self):
        result = full_analysis("test", "critical")
        html = format_html(result)
        assert "<!DOCTYPE html>" in html
        assert "Root Cause Analysis" in html

    def test_json_roundtrip(self):
        result = full_analysis("test", "low")
        dumped = json.dumps(result)
        loaded = json.loads(dumped)
        assert loaded["incident"] == "test"
