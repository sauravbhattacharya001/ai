"""Tests for risk_heatmap module."""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from replication.risk_heatmap import (
    RiskHeatmap, HeatmapConfig, HeatmapResult, RiskItem,
    Likelihood, Impact, RiskCategory,
)


def test_default_generation():
    hm = RiskHeatmap()
    result = hm.generate()
    assert len(result.risks) > 0
    assert '<html' in result.html


def test_custom_config():
    config = HeatmapConfig(agent_count=5, risks_per_agent=2, seed=42)
    result = RiskHeatmap(config).generate()
    assert len(result.risks) == 10  # 5 agents × 2 risks


def test_seed_reproducibility():
    c = HeatmapConfig(seed=123, agent_count=3)
    r1 = RiskHeatmap(c).generate()
    r2 = RiskHeatmap(c).generate()
    assert [r.to_dict() for r in r1.risks] == [r.to_dict() for r in r2.risks]


def test_risk_item_score():
    r = RiskItem(name="test", category="drift", likelihood=3, impact=4)
    assert r.score == 12


def test_risk_item_roundtrip():
    r = RiskItem(name="x", category="evasion", likelihood=2, impact=5, mitigation="fix it", agent_id="a-1")
    d = r.to_dict()
    r2 = RiskItem.from_dict(d)
    assert r2.name == "x"
    assert r2.score == 10
    assert r2.agent_id == "a-1"


def test_html_contains_key_elements():
    result = RiskHeatmap(HeatmapConfig(seed=1)).generate()
    assert 'Risk Heatmap' in result.html
    assert 'ALL_RISKS' in result.html
    assert 'Export JSON' in result.html
    assert 'Catastrophic' in result.html


def test_save_to_file():
    with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
        path = f.name
    try:
        result = RiskHeatmap(HeatmapConfig(seed=1, agent_count=2)).generate()
        result.save(path)
        content = open(path, encoding='utf-8').read()
        assert '<html' in content
    finally:
        os.unlink(path)


def test_json_output():
    result = RiskHeatmap(HeatmapConfig(seed=1, agent_count=3)).generate()
    j = json.loads(result.to_json())
    assert 'risks' in j
    assert 'categories' in j
    assert 'grid' in j
    assert j['risk_count'] == len(result.risks)


def test_import_json():
    risks_data = [
        {"name": "Custom Risk A", "category": "drift", "likelihood": 4, "impact": 3, "mitigation": "m1"},
        {"name": "Custom Risk B", "category": "evasion", "likelihood": 1, "impact": 5},
    ]
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(risks_data, f)
        path = f.name
    try:
        config = HeatmapConfig(import_path=path)
        result = RiskHeatmap(config).generate()
        assert len(result.risks) == 2
        assert result.risks[0].name == "Custom Risk A"
        assert result.risks[1].score == 5
    finally:
        os.unlink(path)


def test_import_json_wrapped():
    """Import from {"risks": [...]} format."""
    risks_data = {"risks": [
        {"name": "R1", "category": "collusion", "likelihood": 3, "impact": 3},
    ]}
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(risks_data, f)
        path = f.name
    try:
        result = RiskHeatmap(HeatmapConfig(import_path=path)).generate()
        assert len(result.risks) == 1
        assert result.risks[0].score == 9
    finally:
        os.unlink(path)


def test_categories_in_generated_risks():
    result = RiskHeatmap(HeatmapConfig(seed=0, agent_count=20, risks_per_agent=5)).generate()
    cats = {r.category for r in result.risks}
    assert len(cats) >= 3  # should have multiple categories


def test_grid_keys():
    result = RiskHeatmap(HeatmapConfig(seed=7, agent_count=5)).generate()
    j = json.loads(result.to_json())
    for key in j['grid']:
        l, i = key.split(',')
        assert 1 <= int(l) <= 5
        assert 1 <= int(i) <= 5


def test_enums():
    assert Likelihood.RARE.value == 1
    assert Impact.CATASTROPHIC.value == 5
    assert RiskCategory.REPLICATION.value == "replication"


def test_empty_import():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump([], f)
        path = f.name
    try:
        result = RiskHeatmap(HeatmapConfig(import_path=path)).generate()
        assert len(result.risks) == 0
        assert '<html' in result.html
    finally:
        os.unlink(path)


def test_single_agent():
    result = RiskHeatmap(HeatmapConfig(agent_count=1, risks_per_agent=1, seed=99)).generate()
    assert len(result.risks) == 1
    assert result.risks[0].agent_id == "agent-0"


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
            print(f"  ✅ {t.__name__}")
        except Exception as e:
            print(f"  ❌ {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} tests passed")
