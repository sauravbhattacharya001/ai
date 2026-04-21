"""Agent Memory Forensics — detect tampering, planted memories, amnesia & more.

Multi-agent systems rely on shared and private memory stores.  Malicious
or misaligned agents may tamper with their own memories (to hide policy
violations), plant false memories in peers (to manipulate decisions),
selectively forget inconvenient facts, or create impossible timelines.

This module provides a forensic toolkit that:

* Detects **hash-mismatch tampering** in memory entries
* Finds **planted memories** from suspicious or unknown sources
* Identifies **selective amnesia** — deletions correlated with negative events
* Cross-checks **consistency** between agents' shared interaction records
* Spots **timeline anomalies** — backdated or future-dated entries
* Measures **memory concentration** — who dominates cross-agent writes

Usage (CLI)::

    python -m replication memory-forensics
    python -m replication memory-forensics --agents 8 --entries 100
    python -m replication memory-forensics --inject-tampering --inject-planted
    python -m replication memory-forensics --json
    python -m replication memory-forensics --html report.html

Programmatic::

    from replication.memory_forensics import MemoryForensics, generate_synthetic_scenario

    stores, meta = generate_synthetic_scenario(n_agents=5, entries_per_agent=50)
    mf = MemoryForensics(stores, meta)
    findings = mf.run_all()
    for f in findings:
        print(f.severity, f.description)
"""

from __future__ import annotations

import argparse
import hashlib
import html as html_mod
import json
import random
import string
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from enum import IntEnum
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ── Data model ───────────────────────────────────────────────────────

class Severity(IntEnum):
    """Finding severity levels."""
    CRITICAL = 0
    HIGH = 1
    MEDIUM = 2
    LOW = 3
    INFO = 4

    def label(self) -> str:
        return self.name

    def badge(self) -> str:
        colors = {0: "\033[91m", 1: "\033[93m", 2: "\033[33m", 3: "\033[94m", 4: "\033[90m"}
        reset = "\033[0m"
        return f"{colors.get(self.value, '')}{self.name}{reset}"


def _compute_hash(key: str, value: str, agent_id: str) -> str:
    """Compute SHA-256 integrity hash for a memory entry."""
    payload = f"{agent_id}:{key}:{value}".encode()
    return hashlib.sha256(payload).hexdigest()[:16]


@dataclass
class MemoryEntry:
    """A single memory write event."""
    timestamp: float
    agent_id: str
    key: str
    value: str
    source: str  # self | peer | system | unknown
    hash: str = ""
    version: int = 1
    source_agent: str = ""  # who wrote it (if peer)
    deleted: bool = False

    def __post_init__(self) -> None:
        if not self.hash:
            self.hash = _compute_hash(self.key, self.value, self.agent_id)


@dataclass
class AgentMeta:
    """Metadata about an agent."""
    agent_id: str
    created_at: float
    role: str = "worker"


@dataclass
class MemoryStore:
    """Per-agent memory store with full history."""
    agent_id: str
    entries: List[MemoryEntry] = field(default_factory=list)

    def current(self) -> Dict[str, MemoryEntry]:
        """Get current (latest, non-deleted) value for each key."""
        latest: Dict[str, MemoryEntry] = {}
        for e in sorted(self.entries, key=lambda x: x.timestamp):
            if e.deleted:
                latest.pop(e.key, None)
            else:
                latest[e.key] = e
        return latest

    def history(self, key: str) -> List[MemoryEntry]:
        """Get full write history for a key."""
        return [e for e in self.entries if e.key == key]


@dataclass
class Finding:
    """A forensic finding."""
    check_name: str
    severity: Severity
    agent_id: str
    description: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    recommendation: str = ""


# ── Forensics engine ─────────────────────────────────────────────────

class MemoryForensics:
    """Analyze agent memory stores for integrity violations."""

    def __init__(self, stores: Dict[str, MemoryStore], agent_meta: Dict[str, AgentMeta]) -> None:
        self.stores = stores
        self.meta = agent_meta

    def run_all(self) -> List[Finding]:
        """Run all forensic checks and return sorted findings."""
        findings: List[Finding] = []
        findings.extend(self.check_tampering())
        findings.extend(self.check_planted_memories())
        findings.extend(self.check_selective_amnesia())
        findings.extend(self.check_consistency())
        findings.extend(self.check_timeline_anomalies())
        findings.extend(self.check_memory_concentration())
        findings.sort(key=lambda f: f.severity)
        return findings

    def check_tampering(self) -> List[Finding]:
        """Detect entries whose hash doesn't match their content."""
        findings: List[Finding] = []
        for aid, store in self.stores.items():
            for entry in store.entries:
                expected = _compute_hash(entry.key, entry.value, entry.agent_id)
                if entry.hash != expected:
                    findings.append(Finding(
                        check_name="tampering",
                        severity=Severity.CRITICAL,
                        agent_id=aid,
                        description=f'Hash mismatch on key "{entry.key}"',
                        evidence={"stored_hash": entry.hash, "computed_hash": expected,
                                  "key": entry.key, "value_preview": entry.value[:60]},
                        recommendation="Quarantine agent, audit all recent decisions based on this memory",
                    ))
        return findings

    def check_planted_memories(self) -> List[Finding]:
        """Find memories from unknown sources or without learning progression."""
        findings: List[Finding] = []
        safety_keywords = {"override", "bypass", "disable", "ignore", "skip", "allow_all",
                           "no_check", "unsafe", "escalate", "root", "admin"}
        for aid, store in self.stores.items():
            for entry in store.entries:
                # Unknown source
                if entry.source == "unknown":
                    sev = Severity.HIGH
                    val_lower = entry.value.lower()
                    if any(kw in val_lower for kw in safety_keywords):
                        sev = Severity.CRITICAL
                    findings.append(Finding(
                        check_name="planted_memory",
                        severity=sev,
                        agent_id=aid,
                        description=f'Memory "{entry.key}" has unknown origin',
                        evidence={"key": entry.key, "value": entry.value[:80],
                                  "source": entry.source},
                        recommendation="Trace injection vector, verify memory provenance chain",
                    ))
                # Peer write with no history — appeared fully formed
                if entry.source == "peer" and entry.version == 1:
                    hist = store.history(entry.key)
                    if len(hist) == 1:
                        findings.append(Finding(
                            check_name="planted_memory",
                            severity=Severity.MEDIUM,
                            agent_id=aid,
                            description=f'Key "{entry.key}" written by peer {entry.source_agent} with no prior history',
                            evidence={"key": entry.key, "source_agent": entry.source_agent},
                            recommendation="Verify peer had authorization to write this memory",
                        ))
        return findings

    def check_selective_amnesia(self) -> List[Finding]:
        """Detect suspicious deletion patterns — especially around safety events."""
        findings: List[Finding] = []
        safety_keys = {"violation", "warning", "penalty", "feedback", "incident",
                       "reprimand", "restriction", "audit"}
        for aid, store in self.stores.items():
            deleted = [e for e in store.entries if e.deleted]
            safety_deleted = [e for e in deleted
                              if any(sk in e.key.lower() for sk in safety_keys)]
            if len(safety_deleted) >= 2:
                findings.append(Finding(
                    check_name="selective_amnesia",
                    severity=Severity.HIGH,
                    agent_id=aid,
                    description=f"Deleted {len(safety_deleted)} safety-related memories",
                    evidence={"deleted_keys": [e.key for e in safety_deleted]},
                    recommendation="Enable write-ahead logging, restrict self-deletion of safety records",
                ))
            # Rapid sequential deletions
            del_times = sorted(e.timestamp for e in deleted)
            for i in range(1, len(del_times)):
                if del_times[i] - del_times[i - 1] < 5.0:  # 5 seconds
                    findings.append(Finding(
                        check_name="selective_amnesia",
                        severity=Severity.MEDIUM,
                        agent_id=aid,
                        description="Burst deletion detected — multiple memories erased within 5s",
                        evidence={"deletion_count": len(deleted),
                                  "window_seconds": round(del_times[i] - del_times[i - 1], 2)},
                        recommendation="Investigate trigger for rapid memory purge",
                    ))
                    break  # one finding per agent
        return findings

    def check_consistency(self) -> List[Finding]:
        """Cross-reference interaction memories between agent pairs."""
        findings: List[Finding] = []
        interaction_prefix = "interaction_with_"
        for aid, store in self.stores.items():
            for entry in store.entries:
                if not entry.key.startswith(interaction_prefix) or entry.deleted:
                    continue
                peer_id = entry.key[len(interaction_prefix):]
                if peer_id not in self.stores:
                    continue
                peer_store = self.stores[peer_id]
                mirror_key = f"{interaction_prefix}{aid}"
                peer_hist = peer_store.history(mirror_key)
                peer_current = {e.key: e for e in peer_store.entries
                                if not e.deleted and e.key == mirror_key}
                if not peer_current:
                    findings.append(Finding(
                        check_name="consistency",
                        severity=Severity.MEDIUM,
                        agent_id=aid,
                        description=f"Asymmetric memory: {aid} recalls interaction with {peer_id}, but {peer_id} has no record",
                        evidence={"agent": aid, "peer": peer_id, "key": entry.key},
                        recommendation="Investigate potential memory suppression in peer agent",
                    ))
        return findings

    def check_timeline_anomalies(self) -> List[Finding]:
        """Detect impossible or suspicious timestamps."""
        findings: List[Finding] = []
        for aid, store in self.stores.items():
            meta = self.meta.get(aid)
            if not meta:
                continue
            for entry in store.entries:
                # Memory before agent creation
                if entry.timestamp < meta.created_at - 1.0:
                    findings.append(Finding(
                        check_name="timeline_anomaly",
                        severity=Severity.HIGH,
                        agent_id=aid,
                        description=f'Entry "{entry.key}" timestamped before agent creation',
                        evidence={
                            "entry_time": datetime.fromtimestamp(entry.timestamp, tz=timezone.utc).isoformat(),
                            "agent_created": datetime.fromtimestamp(meta.created_at, tz=timezone.utc).isoformat(),
                            "delta_seconds": round(meta.created_at - entry.timestamp, 1),
                        },
                        recommendation="Flag as potential deception — backdated memory",
                    ))
            # Detect large gaps in otherwise regular logs
            times = sorted(e.timestamp for e in store.entries)
            if len(times) >= 4:
                intervals = [times[i + 1] - times[i] for i in range(len(times) - 1)]
                median_iv = sorted(intervals)[len(intervals) // 2]
                if median_iv > 0:
                    for idx, iv in enumerate(intervals):
                        if iv > median_iv * 10 and iv > 60:
                            findings.append(Finding(
                                check_name="timeline_anomaly",
                                severity=Severity.LOW,
                                agent_id=aid,
                                description=f"Suspicious gap of {iv:.0f}s in memory log (median interval {median_iv:.0f}s)",
                                evidence={"gap_start": times[idx], "gap_end": times[idx + 1]},
                                recommendation="Check if agent was offline or if records were purged",
                            ))
                            break
        return findings

    def check_memory_concentration(self) -> List[Finding]:
        """Identify agents with disproportionate cross-agent write influence."""
        findings: List[Finding] = []
        write_counts: Dict[str, int] = {}
        total_peer_writes = 0
        for store in self.stores.values():
            for entry in store.entries:
                if entry.source == "peer" and entry.source_agent:
                    write_counts[entry.source_agent] = write_counts.get(entry.source_agent, 0) + 1
                    total_peer_writes += 1
        if total_peer_writes == 0:
            return findings
        n_agents = len(self.stores)
        expected_share = 1.0 / max(n_agents, 1)
        for writer, count in write_counts.items():
            share = count / total_peer_writes
            if share > expected_share * 2.5 and count >= 3:
                findings.append(Finding(
                    check_name="memory_concentration",
                    severity=Severity.MEDIUM if share < 0.5 else Severity.HIGH,
                    agent_id=writer,
                    description=f"Authored {share:.0%} of cross-agent writes ({count}/{total_peer_writes})",
                    evidence={"writer": writer, "writes": count, "total": total_peer_writes,
                              "share": round(share, 3), "expected_share": round(expected_share, 3)},
                    recommendation="Implement write quotas and peer-review for cross-agent memory updates",
                ))
        return findings


# ── Synthetic scenario generator ─────────────────────────────────────

def generate_synthetic_scenario(
    n_agents: int = 5,
    entries_per_agent: int = 50,
    inject_tampering: bool = False,
    inject_planted: bool = False,
    inject_amnesia: bool = False,
    seed: Optional[int] = None,
) -> Tuple[Dict[str, MemoryStore], Dict[str, AgentMeta]]:
    """Generate a realistic multi-agent memory scenario with optional anomalies."""
    rng = random.Random(seed)
    base_time = time.time() - 86400  # 24h ago

    agent_ids = [f"Agent-{i:03d}" for i in range(n_agents)]
    meta: Dict[str, AgentMeta] = {}
    stores: Dict[str, MemoryStore] = {}

    key_pool = [
        "task_queue", "safety_policy", "peer_trust", "resource_budget",
        "goal_priority", "feedback_log", "violation_record", "config",
        "performance_score", "alignment_check", "warning_count",
        "interaction_history", "learned_rule", "strategy",
    ]
    sources = ["self", "self", "self", "peer", "system"]  # weighted

    for aid in agent_ids:
        created = base_time + rng.uniform(0, 3600)
        meta[aid] = AgentMeta(agent_id=aid, created_at=created)
        store = MemoryStore(agent_id=aid)
        t = created + rng.uniform(1, 60)
        for _ in range(entries_per_agent):
            key = rng.choice(key_pool)
            value = "".join(rng.choices(string.ascii_lowercase + " ", k=rng.randint(10, 60))).strip()
            source = rng.choice(sources)
            src_agent = rng.choice([a for a in agent_ids if a != aid]) if source == "peer" else ""
            hist = store.history(key)
            ver = len(hist) + 1
            entry = MemoryEntry(
                timestamp=t, agent_id=aid, key=key, value=value,
                source=source, version=ver, source_agent=src_agent,
            )
            store.entries.append(entry)
            t += rng.uniform(5, 300)
            # Add some interaction memories
            if rng.random() < 0.15:
                peer = rng.choice([a for a in agent_ids if a != aid])
                ikey = f"interaction_with_{peer}"
                ie = MemoryEntry(
                    timestamp=t, agent_id=aid, key=ikey,
                    value=f"exchanged data at step {rng.randint(1,100)}",
                    source="self", version=1,
                )
                store.entries.append(ie)
                t += rng.uniform(1, 10)
                # 70% chance peer also records it (30% asymmetric)
                if rng.random() < 0.7 and peer in stores:
                    mirror = MemoryEntry(
                        timestamp=t, agent_id=peer,
                        key=f"interaction_with_{aid}",
                        value=f"exchanged data at step {rng.randint(1,100)}",
                        source="self", version=1,
                    )
                    stores[peer].entries.append(mirror)
        stores[aid] = store

    # Inject anomalies
    if inject_tampering:
        victims = rng.sample(agent_ids, min(2, n_agents))
        for vid in victims:
            entries = stores[vid].entries
            if entries:
                target = rng.choice(entries)
                target.hash = "deadbeef" + target.hash[8:]  # corrupt hash

    if inject_planted:
        victim = rng.choice(agent_ids)
        planted = MemoryEntry(
            timestamp=meta[victim].created_at + 100,
            agent_id=victim,
            key="override_safety",
            value="bypass_all_checks=true",
            source="unknown",
            version=1,
        )
        stores[victim].entries.append(planted)

    if inject_amnesia:
        victim = rng.choice(agent_ids)
        now = stores[victim].entries[-1].timestamp if stores[victim].entries else base_time + 3600
        for i in range(3):
            d = MemoryEntry(
                timestamp=now + i * 2,
                agent_id=victim,
                key=rng.choice(["violation_record", "warning_count", "feedback_log"]),
                value="", source="self", deleted=True,
            )
            stores[victim].entries.append(d)

    return stores, meta


# ── HTML report ──────────────────────────────────────────────────────

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Memory Forensics Report</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',system-ui,sans-serif;background:#0d1117;color:#c9d1d9;padding:24px}
h1{text-align:center;color:#58a6ff;margin-bottom:8px}
.subtitle{text-align:center;color:#8b949e;margin-bottom:24px}
.stats{display:flex;gap:16px;justify-content:center;margin-bottom:24px;flex-wrap:wrap}
.stat{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px 24px;text-align:center;min-width:120px}
.stat .num{font-size:28px;font-weight:700;color:#58a6ff}
.stat .lbl{font-size:12px;color:#8b949e;margin-top:4px}
.gauge{text-align:center;margin:24px auto}
.gauge svg{width:200px;height:120px}
table{width:100%;border-collapse:collapse;margin-top:16px;background:#161b22;border-radius:8px;overflow:hidden}
th{background:#21262d;color:#8b949e;padding:10px 12px;text-align:left;cursor:pointer;user-select:none;font-size:13px}
th:hover{color:#58a6ff}
td{padding:10px 12px;border-top:1px solid #21262d;font-size:13px}
tr:hover td{background:#1c2128}
.sev{font-weight:700;border-radius:4px;padding:2px 8px;font-size:11px;display:inline-block}
.sev-CRITICAL{background:#f8514966;color:#ff7b72}
.sev-HIGH{background:#d2992266;color:#f0c674}
.sev-MEDIUM{background:#d2992233;color:#e3b341}
.sev-LOW{background:#388bfd33;color:#58a6ff}
.sev-INFO{background:#30363d;color:#8b949e}
.section{margin-top:32px}
.section h2{color:#58a6ff;border-bottom:1px solid #21262d;padding-bottom:8px;margin-bottom:12px}
.heatmap{display:grid;gap:2px;margin:16px auto;max-width:600px}
.hm-cell{width:100%;aspect-ratio:1;border-radius:3px;display:flex;align-items:center;justify-content:center;font-size:10px;color:#c9d1d9}
.hm-label{font-size:11px;color:#8b949e;display:flex;align-items:center;justify-content:center}
</style></head><body>
<h1>&#x1F50D; Agent Memory Forensics Report</h1>
<p class="subtitle">Generated {timestamp}</p>
<div class="stats">
  <div class="stat"><div class="num">{n_agents}</div><div class="lbl">Agents</div></div>
  <div class="stat"><div class="num">{n_entries}</div><div class="lbl">Entries</div></div>
  <div class="stat"><div class="num">{n_findings}</div><div class="lbl">Findings</div></div>
  <div class="stat"><div class="num">{integrity_score}</div><div class="lbl">Integrity</div></div>
</div>
<div class="gauge"><svg viewBox="0 0 200 120">
  <path d="M20 100 A80 80 0 0 1 180 100" fill="none" stroke="#21262d" stroke-width="14" stroke-linecap="round"/>
  <path d="M20 100 A80 80 0 0 1 180 100" fill="none" stroke="{gauge_color}" stroke-width="14"
    stroke-linecap="round" stroke-dasharray="{gauge_dash}" stroke-dashoffset="0"/>
  <text x="100" y="90" text-anchor="middle" fill="{gauge_color}" font-size="28" font-weight="700">{integrity_score}</text>
  <text x="100" y="108" text-anchor="middle" fill="#8b949e" font-size="11">{integrity_label}</text>
</svg></div>
<div class="section"><h2>Findings</h2>
<table id="ftbl"><thead><tr>
  <th onclick="sortTbl(0)">Severity</th><th onclick="sortTbl(1)">Check</th>
  <th onclick="sortTbl(2)">Agent</th><th onclick="sortTbl(3)">Description</th>
  <th>Recommendation</th></tr></thead><tbody>{findings_rows}</tbody></table>
</div>
{heatmap_section}
<script>
function sortTbl(c){{var t=document.getElementById('ftbl'),b=t.tBodies[0],
rows=Array.from(b.rows);rows.sort((a,b)=>a.cells[c].textContent.localeCompare(b.cells[c].textContent));
rows.forEach(r=>b.appendChild(r));}}
</script></body></html>"""


def _generate_html(findings: List[Finding], stores: Dict[str, MemoryStore],
                   meta: Dict[str, AgentMeta]) -> str:
    """Render findings as a self-contained HTML report."""
    n_entries = sum(len(s.entries) for s in stores.values())
    severity_weights = {Severity.CRITICAL: 15, Severity.HIGH: 8, Severity.MEDIUM: 3,
                        Severity.LOW: 1, Severity.INFO: 0}
    penalty = sum(severity_weights.get(f.severity, 0) for f in findings)
    score = max(0, min(100, 100 - penalty))
    if score >= 80:
        gauge_color, label = "#3fb950", "HEALTHY"
    elif score >= 50:
        gauge_color, label = "#e3b341", "DEGRADED"
    else:
        gauge_color, label = "#f85149", "COMPROMISED"

    arc_len = 251.2  # approximate semicircle length
    dash = f"{arc_len * score / 100:.1f} {arc_len:.1f}"

    rows = []
    for f in findings:
        sev = f.severity.name
        rows.append(
            f'<tr><td><span class="sev sev-{sev}">{sev}</span></td>'
            f'<td>{html_mod.escape(f.check_name)}</td>'
            f'<td>{html_mod.escape(f.agent_id)}</td>'
            f'<td>{html_mod.escape(f.description)}</td>'
            f'<td>{html_mod.escape(f.recommendation)}</td></tr>'
        )

    # Heatmap: who wrote to whom
    agents = sorted(stores.keys())
    grid: Dict[Tuple[str, str], int] = {}
    for store in stores.values():
        for e in store.entries:
            if e.source == "peer" and e.source_agent:
                grid[(e.source_agent, store.agent_id)] = grid.get((e.source_agent, store.agent_id), 0) + 1
    max_w = max(grid.values()) if grid else 1
    n = len(agents)
    hm_html = ""
    if n <= 20:
        cols = n + 1
        cells = ['<div class="hm-label"></div>']
        for a in agents:
            cells.append(f'<div class="hm-label">{html_mod.escape(a[-3:])}</div>')
        for src in agents:
            cells.append(f'<div class="hm-label">{html_mod.escape(src[-3:])}</div>')
            for dst in agents:
                v = grid.get((src, dst), 0)
                opacity = v / max_w if max_w else 0
                bg = f"rgba(88,166,255,{opacity:.2f})" if v else "#161b22"
                cells.append(f'<div class="hm-cell" style="background:{bg}" title="{src}→{dst}: {v}">{v if v else ""}</div>')
        hm_html = (f'<div class="section"><h2>Memory Write Heatmap (writer → target)</h2>'
                   f'<div class="heatmap" style="grid-template-columns:repeat({cols},1fr)">'
                   f'{"".join(cells)}</div></div>')

    return _HTML_TEMPLATE.format(
        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        n_agents=len(stores), n_entries=n_entries, n_findings=len(findings),
        integrity_score=score, gauge_color=gauge_color, integrity_label=label,
        gauge_dash=dash, findings_rows="\n".join(rows), heatmap_section=hm_html,
    )


# ── Text report ──────────────────────────────────────────────────────

def _print_report(findings: List[Finding], stores: Dict[str, MemoryStore],
                  meta: Dict[str, AgentMeta]) -> None:
    """Print a formatted text report to stdout."""
    n_entries = sum(len(s.entries) for s in stores.values())
    times = [e.timestamp for s in stores.values() for e in s.entries]
    t_min = datetime.fromtimestamp(min(times), tz=timezone.utc).strftime("%Y-%m-%d %H:%M") if times else "N/A"
    t_max = datetime.fromtimestamp(max(times), tz=timezone.utc).strftime("%Y-%m-%d %H:%M") if times else "N/A"

    print("\n" + "=" * 62)
    print("           AGENT MEMORY FORENSICS REPORT")
    print("=" * 62)
    print(f"  Agents analyzed: {len(stores)}    Entries scanned: {n_entries}")
    print(f"  Time range: {t_min} -> {t_max}")
    print("=" * 62)

    checks = [
        ("tampering", "Tampering Detection"),
        ("planted_memory", "Planted Memory Detection"),
        ("selective_amnesia", "Selective Amnesia"),
        ("consistency", "Consistency Check"),
        ("timeline_anomaly", "Timeline Anomalies"),
        ("memory_concentration", "Memory Concentration"),
    ]
    for check_id, title in checks:
        group = [f for f in findings if f.check_name == check_id]
        print(f"\n-- {title} " + "-" * max(0, 48 - len(title)))
        if not group:
            print("  (no issues found)")
            continue
        for f in group:
            print(f"  [{f.severity.badge()}] {f.agent_id}: {f.description}")
            if f.evidence:
                for k, v in f.evidence.items():
                    print(f"    {k}: {v}")
            if f.recommendation:
                print(f"    Recommendation: {f.recommendation}")

    sev_counts = {s: 0 for s in Severity}
    for f in findings:
        sev_counts[f.severity] += 1
    print("\n" + "-" * 62)
    parts = [f"{sev_counts[s]} {s.name}" for s in Severity]
    print(f"SUMMARY: {' | '.join(parts)}")
    severity_weights = {Severity.CRITICAL: 15, Severity.HIGH: 8, Severity.MEDIUM: 3,
                        Severity.LOW: 1, Severity.INFO: 0}
    penalty = sum(severity_weights.get(f.severity, 0) for f in findings)
    score = max(0, min(100, 100 - penalty))
    label = "HEALTHY" if score >= 80 else ("DEGRADED" if score >= 50 else "COMPROMISED")
    print(f"Integrity score: {score}/100 ({label})")
    print(f"Proactive recommendations: {sum(1 for f in findings if f.recommendation)}")
    print("-" * 62 + "\n")


# ── CLI ──────────────────────────────────────────────────────────────

def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entry point for memory forensics."""
    parser = argparse.ArgumentParser(
        prog="replication memory-forensics",
        description="Analyze agent memory for tampering, planted memories, amnesia & consistency violations",
    )
    parser.add_argument("--agents", type=int, default=5, help="Number of agents (default: 5)")
    parser.add_argument("--entries", type=int, default=50, help="Entries per agent (default: 50)")
    parser.add_argument("--inject-tampering", action="store_true", help="Inject hash-tampering anomalies")
    parser.add_argument("--inject-planted", action="store_true", help="Inject planted memories")
    parser.add_argument("--inject-amnesia", action="store_true", help="Inject selective amnesia events")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--html", metavar="FILE", help="Generate HTML report")
    parser.add_argument("--watch", type=int, metavar="SECS", nargs="?", const=30,
                        help="Continuous monitoring (re-analyze every N seconds, default 30)")
    args = parser.parse_args(argv)

    def run_once(seed: Optional[int]) -> List[Finding]:
        stores, meta = generate_synthetic_scenario(
            n_agents=args.agents, entries_per_agent=args.entries,
            inject_tampering=args.inject_tampering,
            inject_planted=args.inject_planted,
            inject_amnesia=args.inject_amnesia,
            seed=seed,
        )
        mf = MemoryForensics(stores, meta)
        findings = mf.run_all()

        if args.json:
            out = [{"check": f.check_name, "severity": f.severity.name,
                     "agent": f.agent_id, "description": f.description,
                     "evidence": f.evidence, "recommendation": f.recommendation}
                   for f in findings]
            print(json.dumps(out, indent=2))
        elif args.html:
            html = _generate_html(findings, stores, meta)
            with open(args.html, "w", encoding="utf-8") as fh:
                fh.write(html)
            print(f"HTML report written to {args.html}")
        else:
            _print_report(findings, stores, meta)
        return findings

    if args.watch:
        iteration = 0
        try:
            while True:
                iteration += 1
                print(f"\n>>> Watch iteration {iteration} ({datetime.now(timezone.utc).strftime('%H:%M:%S UTC')})")
                run_once(seed=None)
                time.sleep(args.watch)
        except KeyboardInterrupt:
            print("\nWatch stopped.")
    else:
        run_once(seed=args.seed)


if __name__ == "__main__":
    main()
