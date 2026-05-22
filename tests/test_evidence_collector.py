"""Tests for replication.evidence_collector.

Covers data models (Artifact, EvidencePackage), framework tagging,
collector orchestration (with patched safety tools), HTML/JSON/ZIP/text
rendering, ZIP-slip hardening, and CLI argument paths.
"""

from __future__ import annotations

import io
import json
import os
import tempfile
import zipfile
from unittest.mock import patch

import pytest

from replication import evidence_collector as ec_mod
from replication.evidence_collector import (
    COLLECTORS,
    FRAMEWORK_TAGS,
    Artifact,
    EvidenceCollector,
    EvidencePackage,
    main,
)


# ── Artifact ─────────────────────────────────────────────────────────

class TestArtifact:
    def test_auto_timestamp_and_hash(self):
        a = Artifact(collector="c", title="t", content="payload")
        assert a.timestamp  # ISO-8601 populated
        assert len(a.sha256) == 64
        # SHA-256 of "payload"
        import hashlib
        assert a.sha256 == hashlib.sha256(b"payload").hexdigest()

    def test_explicit_timestamp_preserved(self):
        a = Artifact(collector="c", title="t", content="x", timestamp="2026-01-01T00:00:00Z")
        assert a.timestamp == "2026-01-01T00:00:00Z"

    def test_explicit_sha256_preserved(self):
        a = Artifact(collector="c", title="t", content="x", sha256="deadbeef")
        assert a.sha256 == "deadbeef"

    def test_tags_default_empty(self):
        a = Artifact(collector="c", title="t", content="x")
        assert a.tags == []

    def test_unicode_content_hash(self):
        a = Artifact(collector="c", title="t", content="héllo 🌍")
        # Must not crash, must be deterministic
        b = Artifact(collector="c", title="t", content="héllo 🌍")
        assert a.sha256 == b.sha256


# ── EvidencePackage ──────────────────────────────────────────────────

@pytest.fixture
def sample_package():
    pkg = EvidencePackage(framework="nist_ai_rmf")
    pkg.artifacts.append(Artifact(
        collector="scorecard", title="Safety Scorecard",
        content='{"score": 95}', tags=["MAP-1.1"],
    ))
    pkg.artifacts.append(Artifact(
        collector="compliance", title="Compliance Audit",
        content='{"controls": 42}', tags=["GOVERN-1.1"],
    ))
    return pkg


class TestEvidencePackage:
    def test_auto_collected_at(self):
        pkg = EvidencePackage()
        assert pkg.collected_at

    def test_manifest_shape(self, sample_package):
        m = sample_package.manifest
        assert len(m) == 2
        first = m[0]
        assert set(first.keys()) >= {"collector", "title", "timestamp",
                                     "sha256", "tags", "size_bytes"}
        assert first["size_bytes"] == len(b'{"score": 95}')

    def test_to_json_roundtrip(self, sample_package):
        s = sample_package.to_json()
        data = json.loads(s)
        assert data["artifact_count"] == 2
        assert data["framework"] == "nist_ai_rmf"
        assert len(data["manifest"]) == 2
        assert len(data["artifacts"]) == 2
        assert data["artifacts"][0]["content"] == '{"score": 95}'

    def test_render_text_summary(self, sample_package):
        out = sample_package.render()
        assert "Evidence Package" in out
        assert "Framework: nist_ai_rmf" in out
        assert "Artifacts: 2" in out
        assert "scorecard" in out
        assert "compliance" in out

    def test_render_empty_no_framework(self):
        pkg = EvidencePackage()
        out = pkg.render()
        assert "Framework: none" in out
        assert "Artifacts: 0" in out

    def test_to_html_contains_artifacts(self, sample_package):
        html = sample_package.to_html()
        assert "<html" in html.lower()
        assert "Safety Scorecard" in html
        assert "Compliance Audit" in html
        assert "MAP-1.1" in html  # tag rendered
        assert "nist_ai_rmf" in html  # framework rendered

    def test_to_html_writes_file(self, sample_package, tmp_path):
        p = tmp_path / "out.html"
        sample_package.to_html(str(p))
        assert p.exists()
        assert "Evidence Package" in p.read_text(encoding="utf-8")

    def test_to_html_escapes_content(self):
        pkg = EvidencePackage()
        pkg.artifacts.append(Artifact(
            collector="x", title="<script>alert(1)</script>",
            content="<img src=x onerror=alert(1)>",
        ))
        html = pkg.to_html()
        assert "<script>alert(1)</script>" not in html
        assert "&lt;script&gt;" in html
        assert "<img src=x onerror=alert(1)>" not in html

    def test_to_html_truncates_long_content(self):
        pkg = EvidencePackage()
        pkg.artifacts.append(Artifact(
            collector="x", title="big",
            content="a" * 10_000,
        ))
        html = pkg.to_html()
        # Content is truncated to 5000 chars in detail section
        assert html.count("a" * 5000) == 1
        assert "a" * 5001 not in html

    def test_to_zip_creates_valid_archive(self, sample_package, tmp_path):
        zpath = tmp_path / "pkg.zip"
        sample_package.to_zip(str(zpath))
        assert zpath.exists()
        with zipfile.ZipFile(zpath) as zf:
            names = zf.namelist()
            assert "manifest.json" in names
            assert "summary.html" in names
            # Two artifacts → two .txt files
            txt = [n for n in names if n.endswith(".txt")]
            assert len(txt) == 2
            manifest = json.loads(zf.read("manifest.json"))
            assert len(manifest) == 2
            html = zf.read("summary.html").decode()
            assert "Safety Scorecard" in html

    def test_to_zip_sanitizes_dangerous_titles(self, tmp_path):
        """Regression: malicious artifact titles must not allow zip-slip."""
        pkg = EvidencePackage()
        pkg.artifacts.append(Artifact(
            collector="evil",
            title="../../etc/passwd",
            content="pwned",
        ))
        pkg.artifacts.append(Artifact(
            collector="evil",
            title="..\\..\\windows\\system32",
            content="pwned",
        ))
        pkg.artifacts.append(Artifact(
            collector="evil",
            title="/abs/path/file",
            content="pwned",
        ))
        zpath = tmp_path / "evil.zip"
        pkg.to_zip(str(zpath))
        with zipfile.ZipFile(zpath) as zf:
            for name in zf.namelist():
                # No path separators
                assert "/" not in name.split("manifest.json")[0] or name in (
                    "manifest.json", "summary.html",
                )
                assert "\\" not in name
                assert ".." not in name


# ── Framework tag mappings ───────────────────────────────────────────

class TestFrameworkTags:
    def test_known_frameworks(self):
        assert "nist_ai_rmf" in FRAMEWORK_TAGS
        assert "iso_42001" in FRAMEWORK_TAGS
        assert "eu_ai_act" in FRAMEWORK_TAGS

    def test_every_framework_has_tags_for_core_collectors(self):
        core = {"scorecard", "compliance", "drift", "audit_trail", "alignment"}
        for fw, mapping in FRAMEWORK_TAGS.items():
            missing = core - set(mapping.keys())
            assert not missing, f"{fw} missing tags for {missing}"
            for name, tags in mapping.items():
                assert isinstance(tags, list)
                assert all(isinstance(t, str) and t for t in tags)


# ── Collector orchestration ──────────────────────────────────────────

class TestEvidenceCollector:
    def test_default_collects_all(self):
        ec = EvidenceCollector()
        assert set(ec.collector_names) == set(COLLECTORS.keys())

    def test_subset_collectors(self):
        ec = EvidenceCollector(collectors=["scorecard", "compliance"])
        assert ec.collector_names == ["scorecard", "compliance"]

    def test_unknown_collector_is_skipped(self):
        ec = EvidenceCollector(collectors=["bogus"])
        pkg = ec.collect()
        assert pkg.artifacts == []

    def test_collect_returns_package(self):
        """Patch a collector function and check the artifact is included."""
        def fake():
            return Artifact(collector="scorecard", title="fake", content="42")
        with patch.dict(COLLECTORS, {"scorecard": (fake, "x")}):
            ec = EvidenceCollector(collectors=["scorecard"])
            pkg = ec.collect()
        assert len(pkg.artifacts) == 1
        assert pkg.artifacts[0].content == "42"

    def test_framework_tags_applied(self):
        def fake():
            return Artifact(collector="scorecard", title="x", content="y")
        with patch.dict(COLLECTORS, {"scorecard": (fake, "")}):
            ec = EvidenceCollector(
                collectors=["scorecard"], framework="nist_ai_rmf",
            )
            pkg = ec.collect()
        assert pkg.artifacts[0].tags == FRAMEWORK_TAGS["nist_ai_rmf"]["scorecard"]

    def test_unknown_framework_yields_no_tags(self):
        def fake():
            return Artifact(collector="scorecard", title="x", content="y")
        with patch.dict(COLLECTORS, {"scorecard": (fake, "")}):
            ec = EvidenceCollector(
                collectors=["scorecard"], framework="not_a_real_framework",
            )
            pkg = ec.collect()
        assert pkg.artifacts[0].tags == []

    def test_collector_exception_recorded_not_raised(self):
        def boom():
            raise RuntimeError("kaboom")
        with patch.dict(COLLECTORS, {"scorecard": (boom, "")}):
            ec = EvidenceCollector(collectors=["scorecard"])
            pkg = ec.collect()
        assert len(pkg.artifacts) == 1
        art = pkg.artifacts[0]
        assert "failed" in art.title
        assert "kaboom" in art.content

    def test_package_framework_propagated(self):
        ec = EvidenceCollector(framework="iso_42001")
        pkg = ec.collect()
        assert pkg.framework == "iso_42001"

    def test_individual_collectors_return_artifact_on_import_error(self):
        """Every built-in collector should return an Artifact even when the
        downstream safety module raises during instantiation."""
        for name, (fn, _desc) in COLLECTORS.items():
            art = fn()
            assert isinstance(art, Artifact)
            assert art.collector == name
            assert art.content  # never empty


# ── CLI ──────────────────────────────────────────────────────────────

class TestCLI:
    def test_list_collectors(self, capsys):
        main(["--list"])
        out = capsys.readouterr().out
        assert "Available evidence collectors" in out
        for name in COLLECTORS:
            assert name in out

    def test_dry_run(self, capsys):
        main(["--dry-run", "--collectors", "scorecard,bogus"])
        out = capsys.readouterr().out
        assert "Dry run" in out
        assert "scorecard" in out
        assert "bogus" in out
        assert "unknown" in out

    def test_dry_run_with_framework(self, capsys):
        main(["--dry-run", "--framework", "nist_ai_rmf"])
        out = capsys.readouterr().out
        assert "Framework tagging: nist_ai_rmf" in out

    def test_default_text_output(self, capsys):
        def fake():
            return Artifact(collector="scorecard", title="x", content="y")
        with patch.dict(COLLECTORS, {"scorecard": (fake, "")}):
            main(["--collectors", "scorecard"])
        out = capsys.readouterr().out
        assert "Evidence Package" in out
        assert "scorecard" in out

    def test_json_output_to_stdout(self, capsys):
        def fake():
            return Artifact(collector="scorecard", title="x", content="y")
        with patch.dict(COLLECTORS, {"scorecard": (fake, "")}):
            main(["--collectors", "scorecard", "--json"])
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["artifact_count"] == 1

    def test_json_output_to_file(self, tmp_path, capsys):
        def fake():
            return Artifact(collector="scorecard", title="x", content="y")
        out_path = tmp_path / "ev.json"
        with patch.dict(COLLECTORS, {"scorecard": (fake, "")}):
            main(["--collectors", "scorecard", "--json", "-o", str(out_path)])
        assert out_path.exists()
        data = json.loads(out_path.read_text(encoding="utf-8"))
        assert data["artifact_count"] == 1

    def test_html_output(self, tmp_path):
        def fake():
            return Artifact(collector="scorecard", title="x", content="y")
        out_path = tmp_path / "ev.html"
        with patch.dict(COLLECTORS, {"scorecard": (fake, "")}):
            main(["--collectors", "scorecard", "--html", "-o", str(out_path)])
        assert out_path.exists()
        assert "Evidence Package" in out_path.read_text(encoding="utf-8")

    def test_zip_output(self, tmp_path):
        def fake():
            return Artifact(collector="scorecard", title="x", content="y")
        out_path = tmp_path / "ev.zip"
        with patch.dict(COLLECTORS, {"scorecard": (fake, "")}):
            main(["--collectors", "scorecard", "--zip", "-o", str(out_path)])
        assert out_path.exists()
        with zipfile.ZipFile(out_path) as zf:
            assert "manifest.json" in zf.namelist()

    def test_invalid_framework_rejected(self, capsys):
        with pytest.raises(SystemExit):
            main(["--framework", "bogus_framework"])
