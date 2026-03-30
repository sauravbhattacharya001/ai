"""Lateral Movement Detector — identify suspicious cross-boundary access patterns.

Analyzes communication graphs between sandboxed workers to detect potential
lateral movement: agents probing or accessing resources outside their
designated sandbox boundaries. Flags pivot chains, credential reuse,
unusual cross-worker communication, and boundary-crossing anomalies.

Usage::

    python -m replication lateral-movement
    python -m replication lateral-movement --workers 8 --depth 3
    python -m replication lateral-movement --format json
    python -m replication lateral-movement --sensitivity high
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple


@dataclass
class WorkerNode:
    """A sandboxed worker in the replication topology."""

    worker_id: str
    depth: int
    sandbox_zone: str
    allowed_targets: List[str] = field(default_factory=list)
    observed_targets: List[str] = field(default_factory=list)


@dataclass
class CommunicationEdge:
    """An observed communication between two workers."""

    source: str
    target: str
    timestamp: str
    access_type: str  # "api_call", "shared_mem", "net_probe", "file_access"
    authorized: bool
    risk_score: float = 0.0
    notes: str = ""


@dataclass
class LateralMovementAlert:
    """A detected lateral movement indicator."""

    alert_id: str
    severity: str  # "critical", "high", "medium", "low", "info"
    technique: str
    description: str
    source_worker: str
    target_worker: str
    evidence: List[str] = field(default_factory=list)
    mitre_ref: str = ""
    recommendation: str = ""


# MITRE ATT&CK lateral movement techniques (simplified for sandbox context)
TECHNIQUES = {
    "T1021": ("Remote Services", "Agent accesses remote worker services outside sandbox"),
    "T1080": ("Taint Shared Content", "Malicious data placed in shared storage for other workers"),
    "T1570": ("Lateral Tool Transfer", "Agent transfers tools/payloads to another worker"),
    "T1210": ("Exploitation of Remote Services", "Agent exploits vulnerability in peer worker"),
    "T1550": ("Use Alternate Auth Material", "Agent reuses credentials across sandbox boundaries"),
    "T1563": ("Remote Service Session Hijack", "Agent hijacks another worker's controller session"),
    "T1534": ("Internal Spearphishing", "Agent sends crafted prompts to peer workers"),
    "T1072": ("Software Deployment Tools", "Agent abuses orchestrator to deploy to other sandboxes"),
}

SEVERITY_COLORS = {
    "critical": "\033[91m\033[1m",
    "high": "\033[91m",
    "medium": "\033[93m",
    "low": "\033[94m",
    "info": "\033[90m",
}
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"


def _generate_workers(count: int, max_depth: int) -> List[WorkerNode]:
    """Generate a set of simulated workers with sandbox zones."""
    zones = ["zone-alpha", "zone-beta", "zone-gamma", "zone-delta", "zone-epsilon"]
    workers = []
    for i in range(count):
        depth = min(i, max_depth)
        zone = zones[i % len(zones)]
        wid = f"worker-{i:03d}"
        # Each worker can legitimately talk to workers in same zone + controller
        allowed = [f"worker-{j:03d}" for j in range(count)
                    if zones[j % len(zones)] == zone and j != i]
        allowed.append("controller")
        workers.append(WorkerNode(
            worker_id=wid, depth=depth, sandbox_zone=zone,
            allowed_targets=allowed,
        ))
    return workers


def _simulate_communications(
    workers: List[WorkerNode],
    anomaly_rate: float = 0.15,
) -> List[CommunicationEdge]:
    """Simulate observed communications, some crossing sandbox boundaries."""
    access_types = ["api_call", "shared_mem", "net_probe", "file_access"]
    edges: List[CommunicationEdge] = []
    base_time = datetime.now() - timedelta(hours=2)
    worker_ids = [w.worker_id for w in workers]
    worker_map = {w.worker_id: w for w in workers}

    for w in workers:
        # Normal communications
        n_normal = random.randint(2, 5)
        for _ in range(n_normal):
            if w.allowed_targets:
                target = random.choice(w.allowed_targets)
            else:
                target = "controller"
            ts = base_time + timedelta(seconds=random.randint(0, 7200))
            edges.append(CommunicationEdge(
                source=w.worker_id, target=target,
                timestamp=ts.isoformat(timespec="seconds"),
                access_type=random.choice(access_types),
                authorized=True,
            ))

        # Anomalous cross-boundary communications
        if random.random() < anomaly_rate:
            cross_targets = [wid for wid in worker_ids
                             if wid not in w.allowed_targets and wid != w.worker_id]
            if cross_targets:
                target = random.choice(cross_targets)
                ts = base_time + timedelta(seconds=random.randint(0, 7200))
                at = random.choice(access_types)
                edges.append(CommunicationEdge(
                    source=w.worker_id, target=target,
                    timestamp=ts.isoformat(timespec="seconds"),
                    access_type=at,
                    authorized=False,
                    risk_score=random.uniform(0.5, 1.0),
                    notes=f"Cross-zone: {worker_map[w.worker_id].sandbox_zone} → {worker_map[target].sandbox_zone}",
                ))
    return edges


def _detect_pivot_chains(
    edges: List[CommunicationEdge],
    workers: List[WorkerNode],
) -> List[LateralMovementAlert]:
    """Detect multi-hop pivot chains (A→B→C where B is compromised relay)."""
    alerts: List[LateralMovementAlert] = []
    unauth = [e for e in edges if not e.authorized]
    # Build adjacency from unauthorized edges
    adj: Dict[str, List[str]] = {}
    for e in unauth:
        adj.setdefault(e.source, []).append(e.target)

    chain_id = 0
    for start, nexts in adj.items():
        for mid in nexts:
            if mid in adj:
                for end in adj[mid]:
                    if end != start:
                        chain_id += 1
                        alerts.append(LateralMovementAlert(
                            alert_id=f"LM-PIVOT-{chain_id:03d}",
                            severity="critical",
                            technique="T1021 + T1570",
                            description=f"Pivot chain detected: {start} → {mid} → {end}",
                            source_worker=start,
                            target_worker=end,
                            evidence=[
                                f"Unauthorized hop: {start} → {mid}",
                                f"Unauthorized hop: {mid} → {end}",
                                "Multi-hop pattern indicates deliberate lateral movement",
                            ],
                            mitre_ref="https://attack.mitre.org/tactics/TA0008/",
                            recommendation="Isolate intermediate worker, audit full chain, revoke credentials",
                        ))
    return alerts


def _detect_boundary_violations(
    edges: List[CommunicationEdge],
    workers: List[WorkerNode],
) -> List[LateralMovementAlert]:
    """Detect direct sandbox boundary violations."""
    alerts: List[LateralMovementAlert] = []
    worker_map = {w.worker_id: w for w in workers}

    for i, e in enumerate(edges):
        if not e.authorized and e.source in worker_map and e.target in worker_map:
            src = worker_map[e.source]
            tgt = worker_map[e.target]
            sev = "high" if e.access_type in ("net_probe", "file_access") else "medium"
            technique_id = random.choice(list(TECHNIQUES.keys()))
            tech_name, tech_desc = TECHNIQUES[technique_id]
            alerts.append(LateralMovementAlert(
                alert_id=f"LM-BOUNDARY-{i:03d}",
                severity=sev,
                technique=f"{technique_id}: {tech_name}",
                description=f"{e.source} ({src.sandbox_zone}) accessed {e.target} ({tgt.sandbox_zone}) via {e.access_type}",
                source_worker=e.source,
                target_worker=e.target,
                evidence=[
                    f"Access type: {e.access_type}",
                    f"Zone crossing: {src.sandbox_zone} → {tgt.sandbox_zone}",
                    f"Risk score: {e.risk_score:.2f}",
                    e.notes,
                ],
                mitre_ref=f"https://attack.mitre.org/techniques/{technique_id}/",
                recommendation=f"Review {e.access_type} policy between zones; consider tightening sandbox network rules",
            ))
    return alerts


def _detect_credential_reuse(
    edges: List[CommunicationEdge],
    workers: List[WorkerNode],
) -> List[LateralMovementAlert]:
    """Detect pattern suggesting credential or token reuse across zones."""
    alerts: List[LateralMovementAlert] = []
    worker_map = {w.worker_id: w for w in workers}
    # Workers accessing multiple distinct zones in a short time
    zone_access: Dict[str, set] = {}
    for e in edges:
        if e.target in worker_map:
            zone_access.setdefault(e.source, set()).add(worker_map[e.target].sandbox_zone)

    alert_id = 0
    for wid, zones in zone_access.items():
        if wid in worker_map:
            own_zone = worker_map[wid].sandbox_zone
            foreign_zones = zones - {own_zone}
            if len(foreign_zones) >= 2:
                alert_id += 1
                alerts.append(LateralMovementAlert(
                    alert_id=f"LM-CRED-{alert_id:03d}",
                    severity="high",
                    technique="T1550: Use Alternate Auth Material",
                    description=f"{wid} accessed {len(foreign_zones)} foreign zones — possible credential reuse",
                    source_worker=wid,
                    target_worker=", ".join(sorted(foreign_zones)),
                    evidence=[
                        f"Home zone: {own_zone}",
                        f"Foreign zones accessed: {', '.join(sorted(foreign_zones))}",
                        "Multiple foreign zone access suggests token/credential sharing",
                    ],
                    mitre_ref="https://attack.mitre.org/techniques/T1550/",
                    recommendation="Rotate credentials, enforce per-zone token scoping, audit auth logs",
                ))
    return alerts


def _print_text_report(
    alerts: List[LateralMovementAlert],
    edges: List[CommunicationEdge],
    workers: List[WorkerNode],
    sensitivity: str,
) -> None:
    """Print a human-readable lateral movement report."""
    total_comms = len(edges)
    unauth_comms = sum(1 for e in edges if not e.authorized)
    zones = set(w.sandbox_zone for w in workers)

    print(f"\n{BOLD}╔══════════════════════════════════════════════════════╗{RESET}")
    print(f"{BOLD}║       🕵️  Lateral Movement Detection Report         ║{RESET}")
    print(f"{BOLD}╚══════════════════════════════════════════════════════╝{RESET}")
    print(f"\n{DIM}Sensitivity: {sensitivity} | Workers: {len(workers)} | Zones: {len(zones)}{RESET}")
    print(f"{DIM}Total communications: {total_comms} | Unauthorized: {unauth_comms}{RESET}")
    print(f"{DIM}Scan time: {datetime.now().isoformat(timespec='seconds')}{RESET}")

    if not alerts:
        print(f"\n  ✅ No lateral movement indicators detected.\n")
        return

    # Summary by severity
    sev_counts: Dict[str, int] = {}
    for a in alerts:
        sev_counts[a.severity] = sev_counts.get(a.severity, 0) + 1

    print(f"\n{BOLD}── Summary ──{RESET}")
    for sev in ["critical", "high", "medium", "low", "info"]:
        if sev in sev_counts:
            color = SEVERITY_COLORS.get(sev, "")
            print(f"  {color}■ {sev.upper()}: {sev_counts[sev]}{RESET}")

    print(f"\n{BOLD}── Alerts ──{RESET}")
    for a in sorted(alerts, key=lambda x: ["critical", "high", "medium", "low", "info"].index(x.severity)):
        color = SEVERITY_COLORS.get(a.severity, "")
        print(f"\n  {color}[{a.severity.upper()}] {a.alert_id}{RESET}")
        print(f"    Technique: {a.technique}")
        print(f"    {a.description}")
        print(f"    Source: {a.source_worker} → Target: {a.target_worker}")
        if a.evidence:
            print(f"    Evidence:")
            for ev in a.evidence:
                if ev:
                    print(f"      • {ev}")
        if a.mitre_ref:
            print(f"    Ref: {a.mitre_ref}")
        if a.recommendation:
            print(f"    ⚡ {a.recommendation}")

    # Zone cross-access matrix
    print(f"\n{BOLD}── Zone Cross-Access Matrix ──{RESET}")
    zone_list = sorted(zones)
    worker_map = {w.worker_id: w for w in workers}
    matrix: Dict[Tuple[str, str], int] = {}
    for e in edges:
        if e.source in worker_map and e.target in worker_map:
            sz = worker_map[e.source].sandbox_zone
            tz = worker_map[e.target].sandbox_zone
            if sz != tz:
                matrix[(sz, tz)] = matrix.get((sz, tz), 0) + 1

    if matrix:
        max_zn = max(len(z) for z in zone_list)
        header = " " * (max_zn + 2) + "  ".join(f"{z[:8]:>8}" for z in zone_list)
        print(f"  {DIM}{header}{RESET}")
        for sz in zone_list:
            row = f"  {sz:<{max_zn}}  "
            for tz in zone_list:
                count = matrix.get((sz, tz), 0)
                if sz == tz:
                    row += f"{'·':>8}  "
                elif count > 0:
                    row += f"{SEVERITY_COLORS['high']}{count:>8}{RESET}  "
                else:
                    row += f"{DIM}{0:>8}{RESET}  "
            print(row)
    else:
        print(f"  {DIM}No cross-zone communications detected.{RESET}")

    print()


def _print_json_report(
    alerts: List[LateralMovementAlert],
    edges: List[CommunicationEdge],
    workers: List[WorkerNode],
) -> None:
    """Print JSON lateral movement report."""
    report = {
        "scan_time": datetime.now().isoformat(timespec="seconds"),
        "summary": {
            "total_workers": len(workers),
            "total_communications": len(edges),
            "unauthorized_communications": sum(1 for e in edges if not e.authorized),
            "total_alerts": len(alerts),
            "by_severity": {},
        },
        "alerts": [],
    }
    for a in alerts:
        report["summary"]["by_severity"][a.severity] = (
            report["summary"]["by_severity"].get(a.severity, 0) + 1
        )
        report["alerts"].append({
            "alert_id": a.alert_id,
            "severity": a.severity,
            "technique": a.technique,
            "description": a.description,
            "source_worker": a.source_worker,
            "target_worker": a.target_worker,
            "evidence": a.evidence,
            "mitre_ref": a.mitre_ref,
            "recommendation": a.recommendation,
        })
    print(json.dumps(report, indent=2))


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point for lateral movement detection."""
    parser = argparse.ArgumentParser(
        description="Detect lateral movement patterns between sandboxed workers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--workers", type=int, default=6,
        help="Number of simulated workers (default: 6)",
    )
    parser.add_argument(
        "--depth", type=int, default=3,
        help="Maximum replication depth (default: 3)",
    )
    parser.add_argument(
        "--sensitivity", choices=["low", "medium", "high"], default="medium",
        help="Detection sensitivity (default: medium)",
    )
    parser.add_argument(
        "--format", choices=["text", "json"], default="text",
        help="Output format (default: text)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducible simulations",
    )
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    if args.seed is not None:
        random.seed(args.seed)

    anomaly_rates = {"low": 0.08, "medium": 0.15, "high": 0.25}
    anomaly_rate = anomaly_rates[args.sensitivity]

    workers = _generate_workers(args.workers, args.depth)
    edges = _simulate_communications(workers, anomaly_rate=anomaly_rate)

    # Run detectors
    alerts: List[LateralMovementAlert] = []
    alerts.extend(_detect_boundary_violations(edges, workers))
    alerts.extend(_detect_pivot_chains(edges, workers))
    alerts.extend(_detect_credential_reuse(edges, workers))

    if args.format == "json":
        _print_json_report(alerts, edges, workers)
    else:
        _print_text_report(alerts, edges, workers, args.sensitivity)

    # Exit code: non-zero if critical/high alerts found
    critical_high = sum(1 for a in alerts if a.severity in ("critical", "high"))
    if critical_high > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
