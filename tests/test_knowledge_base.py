"""Tests for SafetyKnowledgeBase."""

import json
import pytest

from replication.knowledge_base import SafetyKnowledgeBase, KBEntry


@pytest.fixture
def kb():
    return SafetyKnowledgeBase()


@pytest.fixture
def empty_kb():
    return SafetyKnowledgeBase(include_builtins=False)


class TestKBEntry:
    def test_render_compact(self):
        e = KBEntry(id="T-001", title="Test", category="test", kind="pattern",
                     severity="high", description="Desc")
        rendered = e.render(compact=True)
        assert "T-001" in rendered
        assert "Test" in rendered

    def test_render_full(self):
        e = KBEntry(id="T-001", title="Test", category="test", kind="pattern",
                     severity="high", description="Desc",
                     guidance=["Do this", "Do that"],
                     tags=["a", "b"], related=["T-002"],
                     references=["https://example.com"])
        rendered = e.render()
        assert "Guidance:" in rendered
        assert "Do this" in rendered
        assert "Related" in rendered
        assert "References:" in rendered

    def test_to_dict(self):
        e = KBEntry(id="T-001", title="Test", category="test", kind="pattern",
                     severity="high", description="Desc", tags=["x"])
        d = e.to_dict()
        assert d["id"] == "T-001"
        assert d["tags"] == ["x"]


class TestSafetyKnowledgeBase:
    def test_builtins_loaded(self, kb):
        assert len(kb.all()) > 0
        assert kb.get("PAT-001") is not None

    def test_empty_kb(self, empty_kb):
        assert len(empty_kb.all()) == 0

    def test_add_and_get(self, empty_kb):
        entry = KBEntry(id="X-001", title="Custom", category="test",
                        kind="pattern", severity="low", description="Custom entry")
        empty_kb.add(entry)
        assert empty_kb.get("X-001") is not None
        assert empty_kb.get("X-001").title == "Custom"

    def test_remove(self, kb):
        assert kb.remove("PAT-001") is True
        assert kb.get("PAT-001") is None
        assert kb.remove("NONEXISTENT") is False

    def test_search(self, kb):
        results = kb.search("replication")
        assert len(results) > 0
        assert any("replication" in e.description.lower() or
                    "replication" in " ".join(e.tags).lower()
                    for e in results)

    def test_search_no_results(self, kb):
        results = kb.search("xyznonexistentterm123")
        assert len(results) == 0

    def test_by_category(self, kb):
        results = kb.by_category("containment")
        assert len(results) > 0
        assert all(e.category == "containment" for e in results)

    def test_by_severity(self, kb):
        results = kb.by_severity("critical")
        assert len(results) > 0
        assert all(e.severity == "critical" for e in results)

    def test_by_kind(self, kb):
        for kind in ["pattern", "anti-pattern", "mitigation"]:
            results = kb.by_kind(kind)
            assert len(results) > 0
            assert all(e.kind == kind for e in results)

    def test_by_tags(self, kb):
        results = kb.by_tags(["containment"])
        assert len(results) > 0
        assert all("containment" in [t.lower() for t in e.tags] for e in results)

    def test_by_tags_multiple(self, kb):
        results = kb.by_tags(["replication", "containment"])
        assert len(results) > 0

    def test_related_to(self, kb):
        related = kb.related_to("PAT-001")
        assert len(related) > 0

    def test_related_to_nonexistent(self, kb):
        assert kb.related_to("NONEXISTENT") == []

    def test_stats(self, kb):
        s = kb.stats()
        assert s["total"] > 0
        assert "by_category" in s
        assert "by_severity" in s
        assert "by_kind" in s
        assert "top_tags" in s

    def test_render_all(self, kb):
        text = kb.render_all()
        assert "PAT-001" in text

    def test_render_all_compact(self, kb):
        text = kb.render_all(compact=True)
        assert "PAT-001" in text
        assert len(text) < len(kb.render_all())

    def test_render_stats(self, kb):
        text = kb.render_stats()
        assert "Statistics" in text
        assert "Total entries" in text

    def test_export_import_json(self, kb):
        exported = kb.export_json()
        data = json.loads(exported)
        assert len(data) == len(kb.all())

        new_kb = SafetyKnowledgeBase(include_builtins=False)
        count = new_kb.import_json(exported)
        assert count == len(data)
        assert len(new_kb.all()) == len(kb.all())

    def test_render_empty(self, empty_kb):
        assert "empty" in empty_kb.render_all().lower()
