"""Tests for capability_catalog module."""

import json
import os
import tempfile
import pytest

from replication.capability_catalog import (
    _new_entry,
    _load,
    _save,
    _find,
    add_capability,
    approve_capability,
    revoke_capability,
    search_capabilities,
    show_stats,
    audit_catalog,
    export_catalog,
    main,
)


@pytest.fixture
def catalog_path(tmp_path):
    return str(tmp_path / "catalog.json")


@pytest.fixture
def sample_entries():
    return [
        _new_entry("code execution", "high", "Run arbitrary code", ["runtime"]),
        _new_entry("web search", "low", "Search the internet", ["network"]),
        _new_entry("file write", "critical", "Write to filesystem", ["io", "filesystem"]),
    ]


class TestNewEntry:
    def test_creates_valid_entry(self):
        e = _new_entry("test cap", "high", "desc", ["tag1"])
        assert e["name"] == "test cap"
        assert e["risk"] == "high"
        assert e["status"] == "pending"
        assert e["tags"] == ["tag1"]
        assert e["approved_by"] is None

    def test_defaults(self):
        e = _new_entry("minimal")
        assert e["risk"] == "medium"
        assert e["tags"] == []


class TestLoadSave:
    def test_round_trip(self, catalog_path, sample_entries):
        _save(sample_entries, catalog_path)
        loaded = _load(catalog_path)
        assert len(loaded) == 3
        assert loaded[0]["name"] == "code execution"

    def test_load_missing_file(self, catalog_path):
        assert _load(catalog_path) == []


class TestFind:
    def test_finds_by_name(self, sample_entries):
        assert _find(sample_entries, "code execution") is not None
        assert _find(sample_entries, "CODE EXECUTION") is not None

    def test_not_found(self, sample_entries):
        assert _find(sample_entries, "nonexistent") is None


class TestAddCapability:
    def test_add_new(self, catalog_path):
        entries = []
        add_capability(entries, "new cap", "low", "desc", ["t"], catalog_path)
        assert len(entries) == 1

    def test_duplicate_rejected(self, catalog_path, sample_entries, capsys):
        _save(sample_entries, catalog_path)
        add_capability(sample_entries, "code execution", "low", "", [], catalog_path)
        out = capsys.readouterr().out
        assert "already exists" in out


class TestApproveRevoke:
    def test_approve(self, catalog_path, sample_entries):
        _save(sample_entries, catalog_path)
        approve_capability(sample_entries, "web search", "admin", catalog_path)
        e = _find(sample_entries, "web search")
        assert e["status"] == "approved"
        assert e["approved_by"] == "admin"

    def test_revoke(self, catalog_path, sample_entries):
        _save(sample_entries, catalog_path)
        revoke_capability(sample_entries, "code execution", "too dangerous", catalog_path)
        e = _find(sample_entries, "code execution")
        assert e["status"] == "revoked"
        assert len(e["notes"]) == 1


class TestSearch:
    def test_search_by_name(self, sample_entries, capsys):
        search_capabilities(sample_entries, "web")
        out = capsys.readouterr().out
        assert "web search" in out

    def test_search_by_tag(self, sample_entries, capsys):
        search_capabilities(sample_entries, "filesystem")
        out = capsys.readouterr().out
        assert "file write" in out

    def test_no_results(self, sample_entries, capsys):
        search_capabilities(sample_entries, "zzzzz")
        out = capsys.readouterr().out
        assert "No capabilities" in out


class TestStats:
    def test_stats_output(self, sample_entries, capsys):
        show_stats(sample_entries)
        out = capsys.readouterr().out
        assert "Total capabilities: 3" in out


class TestAudit:
    def test_audit_flags_issues(self, capsys):
        entries = [_new_entry("bare", "critical")]
        audit_catalog(entries)
        out = capsys.readouterr().out
        assert "CRITICAL" in out


class TestExport:
    def test_export_json(self, catalog_path, sample_entries):
        export_catalog(sample_entries, catalog_path)
        with open(catalog_path) as f:
            data = json.load(f)
        assert len(data) == 3


class TestCLI:
    def test_list_empty(self, catalog_path, capsys):
        main(["--catalog", catalog_path, "--list"])
        out = capsys.readouterr().out
        assert "No capabilities" in out

    def test_add_and_list(self, catalog_path, capsys):
        main(["--catalog", catalog_path, "--add", "test", "--risk", "high"])
        main(["--catalog", catalog_path, "--list"])
        out = capsys.readouterr().out
        assert "test" in out

    def test_stats_empty(self, catalog_path, capsys):
        main(["--catalog", catalog_path, "--stats"])
        out = capsys.readouterr().out
        assert "empty" in out
