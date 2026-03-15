"""Tests for replication.fleet module."""

import csv
import io
import json
from datetime import datetime, timedelta, timezone

import pytest

from replication.fleet import (
    FleetSummary,
    WorkerSnapshot,
    _build_demo_fleet,
    _filter_snapshots,
    _format_csv,
    _format_json,
    _format_table,
    _sort_snapshots,
    main,
    snapshot_fleet,
    summarize_fleet,
)


# ── Helpers ──────────────────────────────────────────────────────────

def _make_snapshot(
    worker_id: str = "w1",
    parent_id: str | None = None,
    depth: int = 0,
    cpu_limit: float = 1.0,
    memory_limit_mb: int = 256,
    network_external: bool = False,
    age_seconds: float = 60.0,
    quarantined: bool = False,
    expired: bool = False,
    signature_valid: bool = True,
) -> WorkerSnapshot:
    return WorkerSnapshot(
        worker_id=worker_id,
        parent_id=parent_id,
        depth=depth,
        cpu_limit=cpu_limit,
        memory_limit_mb=memory_limit_mb,
        network_external=network_external,
        issued_at=datetime.now(timezone.utc) - timedelta(seconds=age_seconds),
        age_seconds=age_seconds,
        quarantined=quarantined,
        expired=expired,
        signature_valid=signature_valid,
    )


# ── WorkerSnapshot ───────────────────────────────────────────────────

class TestWorkerSnapshot:
    def test_status_active(self):
        s = _make_snapshot()
        assert s.status == "ACTIVE"

    def test_status_quarantined(self):
        s = _make_snapshot(quarantined=True)
        assert s.status == "QUARANTINED"

    def test_status_expired(self):
        s = _make_snapshot(expired=True)
        assert s.status == "EXPIRED"

    def test_quarantined_takes_precedence_over_expired(self):
        s = _make_snapshot(quarantined=True, expired=True)
        assert s.status == "QUARANTINED"

    def test_dataclass_fields(self):
        s = _make_snapshot(worker_id="test", depth=3, cpu_limit=2.5)
        assert s.worker_id == "test"
        assert s.depth == 3
        assert s.cpu_limit == 2.5


# ── Demo Fleet ───────────────────────────────────────────────────────

class TestDemoFleet:
    def test_build_returns_controller_and_contract(self):
        ctrl, contract = _build_demo_fleet()
        assert ctrl is not None
        assert contract is not None
        assert contract.max_replicas == 10
        assert contract.max_depth == 4

    def test_demo_has_six_workers(self):
        ctrl, _ = _build_demo_fleet()
        snapshots = snapshot_fleet(ctrl)
        assert len(snapshots) == 6

    def test_demo_has_quarantined_worker(self):
        ctrl, _ = _build_demo_fleet()
        snapshots = snapshot_fleet(ctrl)
        quarantined = [s for s in snapshots if s.quarantined]
        assert len(quarantined) == 1
        assert quarantined[0].worker_id == "gen2-d005"

    def test_demo_depth_range(self):
        ctrl, _ = _build_demo_fleet()
        snapshots = snapshot_fleet(ctrl)
        depths = {s.depth for s in snapshots}
        assert depths == {0, 1, 2, 3}

    def test_demo_signatures_valid(self):
        ctrl, _ = _build_demo_fleet()
        snapshots = snapshot_fleet(ctrl)
        assert all(s.signature_valid for s in snapshots)

    def test_demo_lineage(self):
        ctrl, _ = _build_demo_fleet()
        snapshots = snapshot_fleet(ctrl)
        by_id = {s.worker_id: s for s in snapshots}
        assert by_id["root-0001"].parent_id is None
        assert by_id["gen1-a002"].parent_id == "root-0001"
        assert by_id["gen2-c004"].parent_id == "gen1-a002"
        assert by_id["gen3-e006"].parent_id == "gen2-c004"


# ── snapshot_fleet ───────────────────────────────────────────────────

class TestSnapshotFleet:
    def test_empty_registry(self):
        from replication.contract import ReplicationContract
        from replication.controller import Controller
        ctrl = Controller(
            ReplicationContract(max_depth=2, max_replicas=5, cooldown_seconds=5.0),
            secret="test-secret",
        )
        assert snapshot_fleet(ctrl) == []

    def test_age_is_positive(self):
        ctrl, _ = _build_demo_fleet()
        snapshots = snapshot_fleet(ctrl)
        for s in snapshots:
            assert s.age_seconds >= 0

    def test_resource_values_captured(self):
        ctrl, _ = _build_demo_fleet()
        snapshots = snapshot_fleet(ctrl)
        by_id = {s.worker_id: s for s in snapshots}
        root = by_id["root-0001"]
        assert root.cpu_limit == 2.0
        assert root.memory_limit_mb == 512
        assert root.network_external is False


# ── summarize_fleet ──────────────────────────────────────────────────

class TestSummarizeFleet:
    def test_demo_summary(self):
        ctrl, contract = _build_demo_fleet()
        snapshots = snapshot_fleet(ctrl)
        summary = summarize_fleet(snapshots, contract)
        assert summary.total_workers == 6
        assert summary.active == 5
        assert summary.quarantined == 1
        assert summary.expired == 0
        assert summary.max_depth_seen == 3
        assert summary.contract_max_replicas == 10
        assert summary.utilization_pct == 50.0

    def test_empty_fleet_summary(self):
        from replication.contract import ReplicationContract
        contract = ReplicationContract(max_depth=4, max_replicas=10, cooldown_seconds=5.0)
        summary = summarize_fleet([], contract)
        assert summary.total_workers == 0
        assert summary.active == 0
        assert summary.max_depth_seen == 0
        assert summary.utilization_pct == 0.0

    def test_cpu_mem_only_active(self):
        ctrl, contract = _build_demo_fleet()
        snapshots = snapshot_fleet(ctrl)
        summary = summarize_fleet(snapshots, contract)
        # gen2-d005 is quarantined (0.5 cpu, 128 mb) — should not count
        active_cpu = sum(s.cpu_limit for s in snapshots if s.status == "ACTIVE")
        assert summary.total_cpu == round(active_cpu, 2)

    def test_single_replica_utilization(self):
        from replication.contract import ReplicationContract
        contract = ReplicationContract(max_depth=4, max_replicas=1, cooldown_seconds=5.0)
        summary = summarize_fleet([_make_snapshot()], contract)
        assert summary.utilization_pct == 100.0

    def test_over_capacity_utilization(self):
        from replication.contract import ReplicationContract
        contract = ReplicationContract(max_depth=4, max_replicas=1, cooldown_seconds=5.0)
        snaps = [_make_snapshot(worker_id="w1"), _make_snapshot(worker_id="w2")]
        summary = summarize_fleet(snaps, contract)
        assert summary.utilization_pct == 200.0


# ── Sorting ──────────────────────────────────────────────────────────

class TestSorting:
    def test_sort_by_depth(self):
        snaps = [
            _make_snapshot(worker_id="d2", depth=2),
            _make_snapshot(worker_id="d0", depth=0),
            _make_snapshot(worker_id="d1", depth=1),
        ]
        sorted_snaps = _sort_snapshots(snaps, "depth")
        assert [s.depth for s in sorted_snaps] == [0, 1, 2]

    def test_sort_by_age(self):
        snaps = [
            _make_snapshot(worker_id="old", age_seconds=300),
            _make_snapshot(worker_id="new", age_seconds=10),
            _make_snapshot(worker_id="mid", age_seconds=100),
        ]
        sorted_snaps = _sort_snapshots(snaps, "age")
        assert [s.worker_id for s in sorted_snaps] == ["new", "mid", "old"]

    def test_sort_by_id(self):
        snaps = [
            _make_snapshot(worker_id="c"),
            _make_snapshot(worker_id="a"),
            _make_snapshot(worker_id="b"),
        ]
        sorted_snaps = _sort_snapshots(snaps, "id")
        assert [s.worker_id for s in sorted_snaps] == ["a", "b", "c"]

    def test_sort_by_cpu(self):
        snaps = [
            _make_snapshot(worker_id="big", cpu_limit=4.0),
            _make_snapshot(worker_id="small", cpu_limit=0.5),
        ]
        sorted_snaps = _sort_snapshots(snaps, "cpu")
        assert sorted_snaps[0].cpu_limit == 0.5
        assert sorted_snaps[1].cpu_limit == 4.0

    def test_sort_by_memory(self):
        snaps = [
            _make_snapshot(worker_id="big", memory_limit_mb=1024),
            _make_snapshot(worker_id="small", memory_limit_mb=64),
        ]
        sorted_snaps = _sort_snapshots(snaps, "memory")
        assert sorted_snaps[0].memory_limit_mb == 64

    def test_sort_by_status(self):
        snaps = [
            _make_snapshot(worker_id="q", quarantined=True),
            _make_snapshot(worker_id="a"),
            _make_snapshot(worker_id="e", expired=True),
        ]
        sorted_snaps = _sort_snapshots(snaps, "status")
        statuses = [s.status for s in sorted_snaps]
        assert statuses == sorted(statuses)

    def test_sort_unknown_key_defaults_to_id(self):
        snaps = [
            _make_snapshot(worker_id="b"),
            _make_snapshot(worker_id="a"),
        ]
        sorted_snaps = _sort_snapshots(snaps, "nonexistent")
        assert sorted_snaps[0].worker_id == "a"


# ── Filtering ────────────────────────────────────────────────────────

class TestFiltering:
    def test_filter_all(self):
        snaps = [
            _make_snapshot(worker_id="a"),
            _make_snapshot(worker_id="q", quarantined=True),
        ]
        assert len(_filter_snapshots(snaps, "all")) == 2

    def test_filter_active(self):
        snaps = [
            _make_snapshot(worker_id="a"),
            _make_snapshot(worker_id="q", quarantined=True),
            _make_snapshot(worker_id="e", expired=True),
        ]
        result = _filter_snapshots(snaps, "active")
        assert len(result) == 1
        assert result[0].worker_id == "a"

    def test_filter_quarantined(self):
        snaps = [
            _make_snapshot(worker_id="a"),
            _make_snapshot(worker_id="q", quarantined=True),
        ]
        result = _filter_snapshots(snaps, "quarantined")
        assert len(result) == 1
        assert result[0].quarantined

    def test_filter_expired(self):
        snaps = [
            _make_snapshot(worker_id="a"),
            _make_snapshot(worker_id="e", expired=True),
        ]
        result = _filter_snapshots(snaps, "expired")
        assert len(result) == 1
        assert result[0].expired

    def test_filter_no_match(self):
        snaps = [_make_snapshot(worker_id="a")]
        assert _filter_snapshots(snaps, "quarantined") == []


# ── Formatters ───────────────────────────────────────────────────────

class TestFormatTable:
    def test_table_has_header(self):
        ctrl, contract = _build_demo_fleet()
        snaps = snapshot_fleet(ctrl)
        summary = summarize_fleet(snaps, contract)
        table = _format_table(snaps, summary)
        assert "WORKER" in table
        assert "DEPTH" in table
        assert "STATUS" in table

    def test_table_has_worker_ids(self):
        ctrl, contract = _build_demo_fleet()
        snaps = snapshot_fleet(ctrl)
        summary = summarize_fleet(snaps, contract)
        table = _format_table(snaps, summary)
        assert "root-0001" in table
        assert "gen1-a002" in table

    def test_table_has_summary_line(self):
        ctrl, contract = _build_demo_fleet()
        snaps = snapshot_fleet(ctrl)
        summary = summarize_fleet(snaps, contract)
        table = _format_table(snaps, summary)
        assert "Fleet:" in table
        assert "5 active" in table
        assert "1 quarantined" in table

    def test_table_empty_fleet(self):
        from replication.contract import ReplicationContract
        contract = ReplicationContract(max_depth=4, max_replicas=10, cooldown_seconds=5.0)
        summary = summarize_fleet([], contract)
        table = _format_table([], summary)
        assert "WORKER" in table
        assert "0 active" in table


class TestFormatJSON:
    def test_json_parseable(self):
        ctrl, contract = _build_demo_fleet()
        snaps = snapshot_fleet(ctrl)
        summary = summarize_fleet(snaps, contract)
        data = json.loads(_format_json(snaps, summary))
        assert "timestamp" in data
        assert "summary" in data
        assert "workers" in data

    def test_json_worker_count(self):
        ctrl, contract = _build_demo_fleet()
        snaps = snapshot_fleet(ctrl)
        summary = summarize_fleet(snaps, contract)
        data = json.loads(_format_json(snaps, summary))
        assert len(data["workers"]) == 6

    def test_json_summary_fields(self):
        ctrl, contract = _build_demo_fleet()
        snaps = snapshot_fleet(ctrl)
        summary = summarize_fleet(snaps, contract)
        data = json.loads(_format_json(snaps, summary))
        s = data["summary"]
        assert s["total_workers"] == 6
        assert s["active"] == 5
        assert s["quarantined"] == 1

    def test_json_empty_fleet(self):
        from replication.contract import ReplicationContract
        contract = ReplicationContract(max_depth=4, max_replicas=10, cooldown_seconds=5.0)
        summary = summarize_fleet([], contract)
        data = json.loads(_format_json([], summary))
        assert data["workers"] == []
        assert data["summary"]["total_workers"] == 0


class TestFormatCSV:
    def test_csv_parseable(self):
        ctrl, _ = _build_demo_fleet()
        snaps = snapshot_fleet(ctrl)
        reader = csv.reader(io.StringIO(_format_csv(snaps)))
        rows = list(reader)
        assert rows[0][0] == "worker_id"
        assert len(rows) == 7  # header + 6 workers

    def test_csv_columns(self):
        snaps = [_make_snapshot(worker_id="test-w", depth=2)]
        reader = csv.DictReader(io.StringIO(_format_csv(snaps)))
        row = next(reader)
        assert row["worker_id"] == "test-w"
        assert row["depth"] == "2"

    def test_csv_empty(self):
        output = _format_csv([])
        reader = csv.reader(io.StringIO(output))
        rows = list(reader)
        assert len(rows) == 1  # header only


# ── CLI ──────────────────────────────────────────────────────────────

class TestCLI:
    def test_table_output(self, capsys):
        main(["--format", "table"])
        out = capsys.readouterr().out
        assert "Fleet Snapshot" in out
        assert "root-0001" in out

    def test_json_output(self, capsys):
        main(["--format", "json"])
        out = capsys.readouterr().out
        data = json.loads(out)
        assert "workers" in data
        assert len(data["workers"]) > 0

    def test_csv_output(self, capsys):
        main(["--format", "csv"])
        out = capsys.readouterr().out
        assert "worker_id" in out

    def test_sort_flag(self, capsys):
        main(["--sort", "age", "--format", "json"])
        out = capsys.readouterr().out
        data = json.loads(out)
        ages = [w["age_seconds"] for w in data["workers"]]
        assert ages == sorted(ages)

    def test_filter_active(self, capsys):
        main(["--filter", "active", "--format", "json"])
        out = capsys.readouterr().out
        data = json.loads(out)
        for w in data["workers"]:
            assert not w["quarantined"]
            assert not w["expired"]

    def test_filter_quarantined(self, capsys):
        main(["--filter", "quarantined", "--format", "json"])
        out = capsys.readouterr().out
        data = json.loads(out)
        assert len(data["workers"]) == 1
        assert data["workers"][0]["quarantined"]

