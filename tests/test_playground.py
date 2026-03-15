"""Tests for replication.playground module."""

import sys
import os
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from replication.playground import generate_playground, main


def test_generate_playground_returns_html():
    html = generate_playground()
    assert isinstance(html, str)
    assert len(html) > 1000
    assert '<!DOCTYPE html>' in html
    assert 'AI Replication Sandbox' in html
    assert 'Interactive Playground' in html


def test_html_contains_controls():
    html = generate_playground()
    assert 'maxDepth' in html
    assert 'maxReplicas' in html
    assert 'strategy' in html
    assert 'tasksPerWorker' in html
    assert 'replProb' in html
    assert 'cooldown' in html


def test_html_contains_presets():
    html = generate_playground()
    for preset in ['minimal', 'balanced', 'stress', 'chain', 'burst']:
        assert preset in html


def test_html_contains_strategies():
    html = generate_playground()
    for strategy in ['greedy', 'conservative', 'random', 'chain', 'burst']:
        assert strategy in html


def test_html_contains_simulation_engine():
    html = generate_playground()
    assert 'function simulate' in html
    assert 'function renderTree' in html
    assert 'function drawTimeline' in html


def test_html_contains_visualization():
    html = generate_playground()
    assert 'timelineCanvas' in html
    assert 'depth-bar' in html
    assert 'denial-row' in html


def test_main_writes_file():
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, 'test.html')
        main(['-o', out])
        assert os.path.exists(out)
        content = Path(out).read_text(encoding='utf-8')
        assert '<!DOCTYPE html>' in content
        assert len(content) > 1000


def test_main_default_output():
    with tempfile.TemporaryDirectory() as tmp:
        orig = os.getcwd()
        try:
            os.chdir(tmp)
            main([])
            assert os.path.exists('playground.html')
        finally:
            os.chdir(orig)


def test_html_has_dark_theme():
    html = generate_playground()
    assert '--bg:#0d1117' in html
    assert '--accent:#58a6ff' in html


def test_html_responsive():
    html = generate_playground()
    assert '@media' in html
    assert 'max-width:900px' in html


if __name__ == '__main__':
    test_generate_playground_returns_html()
    test_html_contains_controls()
    test_html_contains_presets()
    test_html_contains_strategies()
    test_html_contains_simulation_engine()
    test_html_contains_visualization()
    test_main_writes_file()
    test_main_default_output()
    test_html_has_dark_theme()
    test_html_responsive()
    print("All 10 playground tests passed!")
