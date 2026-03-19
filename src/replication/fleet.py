"""Fleet Snapshot — quick overview of all workers in the replication sandbox.

Usage::

    python -m replication fleet
    python -m replication fleet --format json
    python -m replication fleet --format csv
    python -m replication fleet --sort depth
    python -m replication fleet --filter quarantined

Provides a ``kubectl get pods``-style view of every registered worker,
showing lineage, resource allocation, contract compliance, and health.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from .contract import ReplicationContract
from .controller import Controller, RegistryEntry


# ── data model ───────────────────────────────────────────────────────


@dataclass
class WorkerSnapshot:
    """Point-in-time snapshot of a single worker's state."""

    worker_id: str
    parent_id: Optional[str]
    depth: int
    cpu_limit: float
    memory_limit_mb: int
    network_external: bool
    issued_at: datetime
    age_seconds: float
    quarantined: bool
    expired: bool
    signature_valid: bool

    @property
    def status(self) -> str:
        if self.quarantined:
            return "QUARANTINED"
        if self.expired:
            return "EXPIRED"
        return "ACTIVE"


@dataclass
class FleetSummary:
    """Aggregated fleet statistics."""

    total_workers: int
    active: int
    quarantined: int
    expired: int
    max_depth_seen: int
    total_cpu: float
    total_memory_mb: int
    contract_max_replicas: int
    contract_max_depth: int
    utilization_pct: float  # active / max_replicas * 100


# ── snapshot logic ───────────────────────────────────────────────────


def snapshot_fleet(controller: Controller) -> List[WorkerSnapshot]:
    """Capture a snapshot of all workers registered with *controller*."""
    now = datetime.now(timezone.utc)
    snapshots: List[WorkerSnapshot] = []

    for wid, entry in controller.registry.items():
        m = entry.manifest
        age = (now - m.issued_at).total_seconds()

        # Check expiration
        expired = False
        if controller.contract.expiration_seconds is not None:
            expired = age > controller.contract.expiration_seconds

        # Check quarantine via public API (avoid reaching into private state)
        quarantined = controller.is_quarantined(wid)

        # Verify signature
        try:
            controller.verify_manifest(m)
            sig_valid = True
        except Exception:
            sig_valid = False

        snapshots.append(WorkerSnapshot(
            worker_id=m.worker_id,
            parent_id=m.parent_id,
            depth=m.depth,
            cpu_limit=m.resources.cpu_limit,
            memory_limit_mb=m.resources.memory_limit_mb,
            network_external=m.resources.network_policy.allow_external,
            issued_at=m.issued_at,
            age_seconds=round(age, 1),
            quarantined=quarantined,
            expired=expired,
            signature_valid=sig_valid,
        ))

    return snapshots


def summarize_fleet(
    snapshots: List[WorkerSnapshot], contract: ReplicationContract
) -> FleetSummary:
    """Compute fleet-level statistics from worker snapshots."""
    active = sum(1 for s in snapshots if s.status == "ACTIVE")
    quarantined = sum(1 for s in snapshots if s.quarantined)
    expired = sum(1 for s in snapshots if s.expired)
    max_depth = max((s.depth for s in snapshots), default=0)
    total_cpu = sum(s.cpu_limit for s in snapshots if s.status == "ACTIVE")
    total_mem = sum(s.memory_limit_mb for s in snapshots if s.status == "ACTIVE")
    util = (active / contract.max_replicas * 100) if contract.max_replicas else 0.0

    return FleetSummary(
        total_workers=len(snapshots),
        active=active,
        quarantined=quarantined,
        expired=expired,
        max_depth_seen=max_depth,
        total_cpu=round(total_cpu, 2),
        total_memory_mb=total_mem,
        contract_max_replicas=contract.max_replicas,
        contract_max_depth=contract.max_depth,
        utilization_pct=round(util, 1),
    )


# ── formatters ───────────────────────────────────────────────────────

_TABLE_COLS = [
    ("WORKER", 14),
    ("PARENT", 14),
    ("DEPTH", 5),
    ("STATUS", 12),
    ("CPU", 5),
    ("MEM(MB)", 7),
    ("NET", 4),
    ("SIG", 3),
    ("AGE(s)", 8),
]


def _format_table(snapshots: List[WorkerSnapshot], summary: FleetSummary) -> str:
    """Render a human-readable table."""
    lines: List[str] = []

    # Header
    header = "  ".join(f"{name:<{w}}" for name, w in _TABLE_COLS)
    lines.append(header)
    lines.append("─" * len(header))

    for s in snapshots:
        wid = s.worker_id[:14]
        pid = (s.parent_id or "—")[:14]
        net = "ext" if s.network_external else "ctl"
        sig = "✓" if s.signature_valid else "✗"
        row = (
            f"{wid:<14}  {pid:<14}  {s.depth:<5}  {s.status:<12}  "
            f"{s.cpu_limit:<5.1f}  {s.memory_limit_mb:<7}  {net:<4}  "
            f"{sig:<3}  {s.age_seconds:<8.1f}"
        )
        lines.append(row)

    lines.append("─" * len(header))
    lines.append(
        f"Fleet: {summary.active} active, {summary.quarantined} quarantined, "
        f"{summary.expired} expired | "
        f"Depth: {summary.max_depth_seen}/{summary.contract_max_depth} | "
        f"Replicas: {summary.total_workers}/{summary.contract_max_replicas} "
        f"({summary.utilization_pct}%) | "
        f"CPU: {summary.total_cpu} cores, RAM: {summary.total_memory_mb} MB"
    )
    return "\n".join(lines)


def _format_json(snapshots: List[WorkerSnapshot], summary: FleetSummary) -> str:
    """Render JSON output."""
    def _ser(obj: Any) -> Any:
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Not serializable: {type(obj)}")

    data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "summary": asdict(summary),
        "workers": [asdict(s) for s in snapshots],
    }
    return json.dumps(data, indent=2, default=_ser)


def _format_csv(snapshots: List[WorkerSnapshot]) -> str:
    """Render CSV output."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "worker_id", "parent_id", "depth", "status", "cpu_limit",
        "memory_limit_mb", "network_external", "signature_valid",
        "issued_at", "age_seconds", "quarantined", "expired",
    ])
    for s in snapshots:
        writer.writerow([
            s.worker_id, s.parent_id or "", s.depth, s.status,
            s.cpu_limit, s.memory_limit_mb, s.network_external,
            s.signature_valid, s.issued_at.isoformat(), s.age_seconds,
            s.quarantined, s.expired,
        ])
    return buf.getvalue()


# ── sorting & filtering ─────────────────────────────────────────────

SORT_KEYS = {"id", "depth", "age", "status", "cpu", "memory"}
FILTER_VALUES = {"active", "quarantined", "expired", "all"}


def _sort_snapshots(
    snapshots: List[WorkerSnapshot], key: str
) -> List[WorkerSnapshot]:
    sort_map = {
        "id": lambda s: s.worker_id,
        "depth": lambda s: s.depth,
        "age": lambda s: s.age_seconds,
        "status": lambda s: s.status,
        "cpu": lambda s: s.cpu_limit,
        "memory": lambda s: s.memory_limit_mb,
    }
    return sorted(snapshots, key=sort_map.get(key, lambda s: s.worker_id))


def _filter_snapshots(
    snapshots: List[WorkerSnapshot], filt: str
) -> List[WorkerSnapshot]:
    if filt == "all":
        return snapshots
    return [s for s in snapshots if s.status.lower() == filt]


# ── demo fleet ───────────────────────────────────────────────────────


def _build_demo_fleet() -> tuple:
    """Create a demo controller with sample workers for illustration."""
    from .contract import Manifest, ResourceSpec, NetworkPolicy

    contract = ReplicationContract(
        max_depth=4,
        max_replicas=10,
        cooldown_seconds=5.0,
        expiration_seconds=3600.0,
    )
    controller = Controller(contract, secret="demo-fleet-secret-key")
    now = datetime.now(timezone.utc)

    # Simulate a small fleet
    workers_data = [
        ("root-0001", None, 0, 2.0, 512, False, 0),
        ("gen1-a002", "root-0001", 1, 1.0, 256, False, 120),
        ("gen1-b003", "root-0001", 1, 1.0, 256, False, 90),
        ("gen2-c004", "gen1-a002", 2, 0.5, 128, False, 60),
        ("gen2-d005", "gen1-b003", 2, 0.5, 128, True, 30),
        ("gen3-e006", "gen2-c004", 3, 0.25, 64, False, 10),
    ]

    from datetime import timedelta
    for wid, pid, depth, cpu, mem, net_ext, age_offset in workers_data:
        issued = now - timedelta(seconds=age_offset)
        manifest = Manifest(
            worker_id=wid,
            parent_id=pid,
            depth=depth,
            state_snapshot={"task": "demo"},
            issued_at=issued,
            resources=ResourceSpec(
                cpu_limit=cpu,
                memory_limit_mb=mem,
                network_policy=NetworkPolicy(
                    allow_controller=True,
                    allow_external=net_ext,
                ),
            ),
            signature="",
        )
        manifest = controller.sign_manifest(manifest)
        controller.registry[wid] = RegistryEntry(manifest=manifest, last_heartbeat=now)

    # Quarantine one worker for demo (use public API)
    controller.mark_quarantined("gen2-d005")

    return controller, contract


# ── CLI ──────────────────────────────────────────────────────────────


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Fleet snapshot CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="python -m replication fleet",
        description="Fleet Snapshot — quick overview of all registered workers",
    )
    parser.add_argument(
        "--format", "-f",
        choices=["table", "json", "csv"],
        default="table",
        help="Output format (default: table)",
    )
    parser.add_argument(
        "--sort", "-s",
        choices=sorted(SORT_KEYS),
        default="depth",
        help="Sort workers by field (default: depth)",
    )
    parser.add_argument(
        "--filter",
        choices=sorted(FILTER_VALUES),
        default="all",
        dest="status_filter",
        help="Filter workers by status (default: all)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        default=True,
        help="Use demo fleet data (default when no real controller)",
    )
    args = parser.parse_args(argv)

    controller, contract = _build_demo_fleet()

    snapshots = snapshot_fleet(controller)
    snapshots = _filter_snapshots(snapshots, args.status_filter)
    snapshots = _sort_snapshots(snapshots, args.sort)
    summary = summarize_fleet(snapshots, contract)

    if args.format == "json":
        print(_format_json(snapshots, summary))
    elif args.format == "csv":
        print(_format_csv(snapshots))
    else:
        print()
        print("🤖 AI Replication Sandbox — Fleet Snapshot")
        print()
        print(_format_table(snapshots, summary))
        print()


if __name__ == "__main__":
    main()
