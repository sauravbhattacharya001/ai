"""Tests for blast_radius module."""

from replication.blast_radius import BlastRadiusAnalyzer, BlastResult, ImpactNode


def test_analyze_leaf_control():
    """A control with no dependents has zero blast radius."""
    analyzer = BlastRadiusAnalyzer({
        "base": [],
        "child": ["base"],
    })
    result = analyzer.analyze("child")
    assert len(result.impacted) == 0


def test_analyze_root_control():
    """Failing a root control cascades to dependents."""
    analyzer = BlastRadiusAnalyzer({
        "base": [],
        "child": ["base"],
        "grandchild": ["child"],
    })
    result = analyzer.analyze("base")
    names = {n.name for n in result.impacted}
    assert names == {"child", "grandchild"}
    assert result.max_depth == 2


def test_impact_ratio():
    analyzer = BlastRadiusAnalyzer({
        "a": [],
        "b": ["a"],
        "c": ["a"],
        "d": [],
    })
    result = analyzer.analyze("a")
    assert len(result.impacted) == 2
    assert result.total_controls == 4
    assert abs(result.impact_ratio - 0.5) < 0.01


def test_severity_levels():
    r = BlastResult(failed_control="x", impacted=[], total_controls=10)
    assert r.severity() == "LOW"

    r.impacted = [ImpactNode("a", 1, ["x", "a"]), ImpactNode("b", 1, ["x", "b"])]
    r.total_controls = 10
    assert r.severity() == "MEDIUM"

    r.impacted = [ImpactNode(f"n{i}", 1, ["x", f"n{i}"]) for i in range(4)]
    assert r.severity() == "HIGH"

    r.impacted = [ImpactNode(f"n{i}", 1, ["x", f"n{i}"]) for i in range(7)]
    assert r.severity() == "CRITICAL"


def test_analyze_all():
    analyzer = BlastRadiusAnalyzer({
        "a": [],
        "b": ["a"],
        "c": ["b"],
    })
    results = analyzer.analyze_all()
    assert len(results) == 3
    # first result should be the one with most impact
    assert results[0].failed_control == "a"


def test_add_and_remove_control():
    analyzer = BlastRadiusAnalyzer({"a": [], "b": ["a"]})
    analyzer.add_control("c", depends_on=["b"])
    result = analyzer.analyze("a")
    assert any(n.name == "c" for n in result.impacted)

    analyzer.remove_control("b")
    result = analyzer.analyze("a")
    assert not any(n.name == "b" for n in result.impacted)


def test_summary_output():
    analyzer = BlastRadiusAnalyzer({"a": [], "b": ["a"]})
    result = analyzer.analyze("a")
    text = result.summary()
    assert "Blast Radius" in text
    assert "b" in text


def test_json_export():
    import json
    analyzer = BlastRadiusAnalyzer({"a": [], "b": ["a"]})
    data = json.loads(analyzer.to_json())
    assert len(data) == 2
    assert any(d["control"] == "a" for d in data)


def test_html_export():
    analyzer = BlastRadiusAnalyzer({"a": [], "b": ["a"]})
    html = analyzer.to_html()
    assert "<html" in html
    assert "Blast Radius" in html


def test_default_controls():
    """The default control graph should work without errors."""
    analyzer = BlastRadiusAnalyzer()
    results = analyzer.analyze_all()
    assert len(results) > 0
    # audit_trail should have high impact since many depend on it
    audit = next(r for r in results if r.failed_control == "audit_trail")
    assert len(audit.impacted) > 10
