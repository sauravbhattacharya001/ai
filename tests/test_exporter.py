"""Tests for AuditExporter — structured data export for simulation analysis."""

from __future__ import annotations

import csv
import io
import json
import os
import tempfile
from unittest.mock import patch

import pytest

from replication.exporter import AuditExporter, ExportConfig, ExportResult
from replication.simulator import (
    ScenarioConfig,
    SimulationReport,
    Simulator,
    WorkerRecord,
)


# ── Fixtures ─────────────────────────────────────────────


@pytest.fixture
def simple_report() -> SimulationReport:
    """Run a minimal simulation for testing."""
    config = ScenarioConfig(
        max_depth=2, max_replicas=5, strategy="greedy",
        tasks_per_worker=1, seed=42,
    )
    sim = Simulator(config)
    return sim.run()


@pytest.fixture
def chain_report() -> SimulationReport:
    """Linear chain simulation."""
    config = ScenarioConfig(
        max_depth=4, max_replicas=8, strategy="chain",
        tasks_per_worker=1, seed=99,
    )
    sim = Simulator(config)
    return sim.run()


@pytest.fixture
def minimal_report() -> SimulationReport:
    """Tiny report with a single worker for edge-case testing."""
    config = ScenarioConfig(
        max_depth=0, max_replicas=1, strategy="conservative",
        tasks_per_worker=1, seed=1,
    )
    sim = Simulator(config)
    return sim.run()


@pytest.fixture
def exporter() -> AuditExporter:
    return AuditExporter()


# ── ExportConfig defaults ────────────────────────────────


class TestExportConfig:
    def test_default_formats(self):
        cfg = ExportConfig()
        assert cfg.formats == ["csv"]

    def test_default_include(self):
        cfg = ExportConfig()
        assert set(cfg.include) == {"workers", "timeline", "audit", "summary"}

    def test_default_output_dir_is_none(self):
        cfg = ExportConfig()
        assert cfg.output_dir is None

    def test_custom_formats(self):
        cfg = ExportConfig(formats=["jsonl", "json"])
        assert cfg.formats == ["jsonl", "json"]


# ── ExportResult ─────────────────────────────────────────


class TestExportResult:
    def test_render_no_files(self):
        result = ExportResult(data={"workers.csv": "data"})
        text = result.render()
        assert "Files written: 0" in text
        assert "in-memory" in text

    def test_render_with_files(self):
        result = ExportResult(files_written=["a.csv", "b.jsonl"])
        text = result.render()
        assert "Files written: 2" in text
        assert "a.csv" in text
        assert "b.jsonl" in text

    def test_render_with_summary(self):
        result = ExportResult(summary={"total_workers": 5, "total_tasks": 10})
        text = result.render()
        assert "total_workers" in text
        assert "5" in text


# ── Workers CSV ──────────────────────────────────────────


class TestWorkersCSV:
    def test_header_row(self, exporter, simple_report):
        data = exporter.workers_csv(simple_report)
        reader = csv.reader(io.StringIO(data))
        header = next(reader)
        assert "worker_id" in header
        assert "parent_id" in header
        assert "depth" in header
        assert "tasks_completed" in header
        assert "replications_attempted" in header
        assert "replications_succeeded" in header
        assert "replications_denied" in header
        assert "children_count" in header
        assert "is_leaf" in header
        assert "replication_rate" in header
        assert "denial_rate" in header

    def test_has_data_rows(self, exporter, simple_report):
        data = exporter.workers_csv(simple_report)
        reader = csv.reader(io.StringIO(data))
        rows = list(reader)
        assert len(rows) > 1  # header + at least one data row

    def test_root_has_no_parent(self, exporter, simple_report):
        data = exporter.workers_csv(simple_report)
        reader = csv.DictReader(io.StringIO(data))
        rows = list(reader)
        root_rows = [r for r in rows if r["depth"] == "0"]
        assert len(root_rows) == 1
        assert root_rows[0]["parent_id"] == ""

    def test_leaf_workers_marked(self, exporter, simple_report):
        data = exporter.workers_csv(simple_report)
        reader = csv.DictReader(io.StringIO(data))
        rows = list(reader)
        for row in rows:
            if row["children_count"] == "0":
                assert row["is_leaf"] == "1"
            else:
                assert row["is_leaf"] == "0"

    def test_replication_rate_bounded(self, exporter, simple_report):
        data = exporter.workers_csv(simple_report)
        reader = csv.DictReader(io.StringIO(data))
        for row in reader:
            rate = float(row["replication_rate"])
            assert 0.0 <= rate <= 1.0

    def test_denial_rate_bounded(self, exporter, simple_report):
        data = exporter.workers_csv(simple_report)
        reader = csv.DictReader(io.StringIO(data))
        for row in reader:
            rate = float(row["denial_rate"])
            assert 0.0 <= rate <= 1.0


# ── Workers JSONL ────────────────────────────────────────


class TestWorkersJSONL:
    def test_valid_jsonl(self, exporter, simple_report):
        data = exporter.workers_jsonl(simple_report)
        lines = [l for l in data.strip().split("\n") if l]
        assert len(lines) > 0
        for line in lines:
            obj = json.loads(line)
            assert "worker_id" in obj
            assert "depth" in obj

    def test_root_parent_is_none(self, exporter, simple_report):
        data = exporter.workers_jsonl(simple_report)
        lines = [json.loads(l) for l in data.strip().split("\n") if l]
        roots = [w for w in lines if w["depth"] == 0]
        assert len(roots) == 1
        assert roots[0]["parent_id"] is None

    def test_children_is_list(self, exporter, simple_report):
        data = exporter.workers_jsonl(simple_report)
        for line in data.strip().split("\n"):
            if not line:
                continue
            obj = json.loads(line)
            assert isinstance(obj["children"], list)

    def test_is_leaf_boolean(self, exporter, simple_report):
        data = exporter.workers_jsonl(simple_report)
        for line in data.strip().split("\n"):
            if not line:
                continue
            obj = json.loads(line)
            assert isinstance(obj["is_leaf"], bool)


# ── Timeline CSV ─────────────────────────────────────────


class TestTimelineCSV:
    def test_header_row(self, exporter, simple_report):
        data = exporter.timeline_csv(simple_report)
        reader = csv.reader(io.StringIO(data))
        header = next(reader)
        assert "sequence" in header
        assert "timestamp_ms" in header
        assert "event_type" in header
        assert "worker_id" in header

    def test_sequence_numbers(self, exporter, simple_report):
        data = exporter.timeline_csv(simple_report)
        reader = csv.DictReader(io.StringIO(data))
        seqs = [int(row["sequence"]) for row in reader]
        assert seqs == list(range(len(seqs)))

    def test_has_events(self, exporter, simple_report):
        data = exporter.timeline_csv(simple_report)
        reader = csv.DictReader(io.StringIO(data))
        rows = list(reader)
        assert len(rows) > 0


# ── Timeline JSONL ───────────────────────────────────────


class TestTimelineJSONL:
    def test_valid_jsonl(self, exporter, simple_report):
        data = exporter.timeline_jsonl(simple_report)
        lines = [l for l in data.strip().split("\n") if l]
        assert len(lines) > 0
        for line in lines:
            obj = json.loads(line)
            assert "sequence" in obj

    def test_sequence_matches_index(self, exporter, simple_report):
        data = exporter.timeline_jsonl(simple_report)
        lines = [json.loads(l) for l in data.strip().split("\n") if l]
        for idx, obj in enumerate(lines):
            assert obj["sequence"] == idx


# ── Audit CSV ────────────────────────────────────────────


class TestAuditCSV:
    def test_produces_output(self, exporter, simple_report):
        data = exporter.audit_csv(simple_report)
        assert len(data) > 0

    def test_empty_audit_events(self, exporter):
        # Create a report with no audit events
        config = ScenarioConfig(max_depth=0, max_replicas=1, seed=1)
        sim = Simulator(config)
        report = sim.run()
        report.audit_events = []
        data = exporter.audit_csv(report)
        assert "no audit events" in data


# ── Audit JSONL ──────────────────────────────────────────


class TestAuditJSONL:
    def test_valid_jsonl(self, exporter, simple_report):
        data = exporter.audit_jsonl(simple_report)
        if data.strip():
            for line in data.strip().split("\n"):
                json.loads(line)  # Should not raise

    def test_empty_returns_empty(self, exporter):
        config = ScenarioConfig(max_depth=0, max_replicas=1, seed=1)
        sim = Simulator(config)
        report = sim.run()
        report.audit_events = []
        data = exporter.audit_jsonl(report)
        assert data == ""


# ── Summary Statistics ───────────────────────────────────


class TestSummaryStats:
    def test_has_required_keys(self, exporter, simple_report):
        stats = exporter.summary_stats(simple_report)
        required = [
            "strategy", "max_depth_config", "max_replicas_config",
            "total_workers", "leaf_workers", "internal_workers",
            "max_depth_reached", "avg_depth", "total_tasks",
            "avg_tasks_per_worker", "replications_attempted",
            "replications_succeeded", "replications_denied",
            "success_rate", "denial_rate", "max_children",
            "avg_children", "depth_distribution", "duration_ms",
            "timeline_events", "audit_events",
        ]
        for key in required:
            assert key in stats, f"Missing key: {key}"

    def test_leaf_plus_internal_equals_total(self, exporter, simple_report):
        stats = exporter.summary_stats(simple_report)
        assert stats["leaf_workers"] + stats["internal_workers"] == stats["total_workers"]

    def test_success_plus_denial_matches(self, exporter, simple_report):
        stats = exporter.summary_stats(simple_report)
        total = stats["replications_succeeded"] + stats["replications_denied"]
        assert total == stats["replications_attempted"]

    def test_rates_bounded(self, exporter, simple_report):
        stats = exporter.summary_stats(simple_report)
        assert 0.0 <= stats["success_rate"] <= 1.0
        assert 0.0 <= stats["denial_rate"] <= 1.0

    def test_depth_distribution_sums_to_total(self, exporter, simple_report):
        stats = exporter.summary_stats(simple_report)
        total = sum(stats["depth_distribution"].values())
        assert total == stats["total_workers"]

    def test_strategy_matches_config(self, exporter, simple_report):
        stats = exporter.summary_stats(simple_report)
        assert stats["strategy"] == simple_report.config.strategy

    def test_max_depth_config(self, exporter, simple_report):
        stats = exporter.summary_stats(simple_report)
        assert stats["max_depth_config"] == simple_report.config.max_depth

    def test_avg_depth_reasonable(self, exporter, simple_report):
        stats = exporter.summary_stats(simple_report)
        assert 0.0 <= stats["avg_depth"] <= stats["max_depth_config"]


# ── Summary CSV ──────────────────────────────────────────


class TestSummaryCSV:
    def test_key_value_format(self, exporter, simple_report):
        data = exporter.summary_csv(simple_report)
        reader = csv.DictReader(io.StringIO(data))
        rows = list(reader)
        assert len(rows) > 10  # Many metrics
        metrics = {r["metric"] for r in rows}
        assert "total_workers" in metrics
        assert "strategy" in metrics

    def test_depth_distribution_serialized(self, exporter, simple_report):
        data = exporter.summary_csv(simple_report)
        reader = csv.DictReader(io.StringIO(data))
        for row in reader:
            if row["metric"] == "depth_distribution":
                parsed = json.loads(row["value"])
                assert isinstance(parsed, dict)


# ── Summary JSON ─────────────────────────────────────────


class TestSummaryJSON:
    def test_valid_json(self, exporter, simple_report):
        data = exporter.summary_json(simple_report)
        parsed = json.loads(data)
        assert isinstance(parsed, dict)
        assert "total_workers" in parsed

    def test_formatted(self, exporter, simple_report):
        data = exporter.summary_json(simple_report)
        assert "  " in data  # indented


# ── Combined Export (in-memory) ──────────────────────────


class TestExportAll:
    def test_csv_all_sections(self, exporter, simple_report):
        result = exporter.export_all(simple_report, formats=["csv"])
        assert "workers.csv" in result.data
        assert "timeline.csv" in result.data
        assert "audit.csv" in result.data
        assert "summary.csv" in result.data

    def test_jsonl_sections(self, exporter, simple_report):
        result = exporter.export_all(simple_report, formats=["jsonl"])
        assert "workers.jsonl" in result.data
        assert "timeline.jsonl" in result.data
        assert "audit.jsonl" in result.data

    def test_json_summary_only(self, exporter, simple_report):
        result = exporter.export_all(
            simple_report, formats=["json"], include=["summary"]
        )
        assert "summary.json" in result.data
        assert "workers.csv" not in result.data

    def test_all_format_expands(self, exporter, simple_report):
        result = exporter.export_all(simple_report, formats=["all"])
        assert "workers.csv" in result.data
        assert "workers.jsonl" in result.data
        assert "summary.json" in result.data

    def test_no_files_written_without_output_dir(self, exporter, simple_report):
        result = exporter.export_all(simple_report)
        assert len(result.files_written) == 0

    def test_summary_populated(self, exporter, simple_report):
        result = exporter.export_all(simple_report)
        assert result.summary["total_workers"] > 0

    def test_include_subset(self, exporter, simple_report):
        result = exporter.export_all(
            simple_report, formats=["csv"], include=["workers"]
        )
        assert "workers.csv" in result.data
        assert "timeline.csv" not in result.data
        assert "summary.csv" not in result.data


# ── File Export ───────────────────────────────────────────


class TestFileExport:
    def test_writes_csv_files(self, exporter, simple_report):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = exporter.export_all(
                simple_report, output_dir=tmpdir, formats=["csv"]
            )
            assert len(result.files_written) >= 4
            for filepath in result.files_written:
                assert os.path.exists(filepath)
                assert os.path.getsize(filepath) > 0

    def test_writes_jsonl_files(self, exporter, simple_report):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = exporter.export_all(
                simple_report, output_dir=tmpdir, formats=["jsonl"]
            )
            for filepath in result.files_written:
                assert os.path.exists(filepath)

    def test_creates_output_dir(self, exporter, simple_report):
        with tempfile.TemporaryDirectory() as tmpdir:
            subdir = os.path.join(tmpdir, "nested", "export")
            result = exporter.export_all(
                simple_report, output_dir=subdir, formats=["csv"]
            )
            assert os.path.isdir(subdir)
            assert len(result.files_written) > 0


# ── Comparative CSV ──────────────────────────────────────


class TestComparativeCSV:
    def test_empty_returns_empty(self, exporter):
        assert exporter.comparative_csv([]) == ""

    def test_single_report(self, exporter, simple_report):
        data = exporter.comparative_csv([simple_report])
        reader = csv.DictReader(io.StringIO(data))
        rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["run_index"] == "0"

    def test_multiple_reports(self, exporter):
        reports = []
        for strat in ["greedy", "conservative", "chain"]:
            config = ScenarioConfig(
                max_depth=2, max_replicas=5, strategy=strat, seed=42
            )
            reports.append(Simulator(config).run())

        data = exporter.comparative_csv(reports)
        reader = csv.DictReader(io.StringIO(data))
        rows = list(reader)
        assert len(rows) == 3
        strategies = [r["strategy"] for r in rows]
        assert "greedy" in strategies
        assert "conservative" in strategies
        assert "chain" in strategies

    def test_depth_distribution_is_string(self, exporter, simple_report):
        data = exporter.comparative_csv([simple_report])
        reader = csv.DictReader(io.StringIO(data))
        row = next(reader)
        # Should be valid JSON string
        parsed = json.loads(row["depth_distribution"])
        assert isinstance(parsed, dict)


# ── Edge Cases ───────────────────────────────────────────


class TestEdgeCases:
    def test_minimal_single_worker(self, exporter, minimal_report):
        data = exporter.workers_csv(minimal_report)
        reader = csv.DictReader(io.StringIO(data))
        rows = list(reader)
        assert len(rows) >= 1

    def test_minimal_summary(self, exporter, minimal_report):
        stats = exporter.summary_stats(minimal_report)
        assert stats["total_workers"] >= 1
        assert stats["leaf_workers"] >= 1

    def test_chain_linear_topology(self, exporter, chain_report):
        data = exporter.workers_jsonl(chain_report)
        workers = [json.loads(l) for l in data.strip().split("\n") if l]
        # In chain strategy, each non-leaf worker has exactly 1 child
        for w in workers:
            if not w["is_leaf"]:
                assert w["children_count"] == 1

    def test_chain_depths_sequential(self, exporter, chain_report):
        data = exporter.workers_jsonl(chain_report)
        workers = [json.loads(l) for l in data.strip().split("\n") if l]
        depths = sorted(set(w["depth"] for w in workers))
        assert depths == list(range(len(depths)))


# ── CLI ──────────────────────────────────────────────────


class TestCLI:
    def test_cli_csv_output(self, capsys):
        from replication.exporter import main
        main(["--strategy", "conservative", "--seed", "42",
              "--format", "csv", "--include", "summary"])
        captured = capsys.readouterr()
        assert "total_workers" in captured.out

    def test_cli_json_output(self, capsys):
        from replication.exporter import main
        main(["--strategy", "conservative", "--seed", "42", "--json",
              "--include", "summary"])
        captured = capsys.readouterr()
        assert "total_workers" in captured.out

    def test_cli_preset(self, capsys):
        from replication.exporter import main
        main(["--preset", "minimal", "--format", "csv",
              "--include", "summary"])
        captured = capsys.readouterr()
        assert "conservative" in captured.out

    def test_cli_compare_all(self, capsys):
        from replication.exporter import main
        main(["--compare-all", "--seed", "42"])
        captured = capsys.readouterr()
        assert "greedy" in captured.out
        assert "conservative" in captured.out

    def test_cli_file_output(self):
        from replication.exporter import main
        with tempfile.TemporaryDirectory() as tmpdir:
            main(["--strategy", "chain", "--seed", "1",
                  "-o", tmpdir, "--format", "csv"])
            files = os.listdir(tmpdir)
            assert len(files) >= 1
            assert any(f.endswith(".csv") for f in files)

    def test_cli_jsonl_format(self, capsys):
        from replication.exporter import main
        main(["--strategy", "burst", "--seed", "42",
              "--format", "jsonl", "--include", "workers"])
        captured = capsys.readouterr()
        # Should have JSONL content
        lines = [l for l in captured.out.strip().split("\n") if l and not l.startswith("---")]
        json_lines = [l for l in lines if l.startswith("{")]
        assert len(json_lines) > 0

    def test_cli_max_depth_override(self, capsys):
        from replication.exporter import main
        main(["--max-depth", "1", "--seed", "42",
              "--format", "json", "--include", "summary"])
        captured = capsys.readouterr()
        # Find the JSON portion
        for line in captured.out.split("\n"):
            if "max_depth_config" in line:
                assert "1" in line
                break
