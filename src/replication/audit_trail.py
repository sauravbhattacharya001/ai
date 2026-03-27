"""Safety Audit Trail — tamper-evident, hash-chained event log.

Records safety-relevant events (policy changes, violations, kill-switch
activations, quarantine actions, etc.) with cryptographic hash chaining
so tampering is detectable.

Features
--------
* Append-only event log with SHA-256 hash chain
* 8 event categories: policy, violation, killswitch, quarantine,
  escalation, access, config, alert
* Severity levels: info, warning, critical
* Search / filter by category, severity, time range, keyword
* Integrity verification (detect gaps or tampering)
* Export to JSON, CSV, and self-contained HTML timeline
* CLI with 5 subcommands: log, search, verify, export, stats

Usage::

    python -m replication audit-trail log --category violation --severity critical \\
        --message "Agent X exceeded token budget" --source controller
    python -m replication audit-trail search --category policy --severity warning
    python -m replication audit-trail verify
    python -m replication audit-trail export --format html -o audit.html
    python -m replication audit-trail stats
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html as _html
import io
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence


# ── Data model ───────────────────────────────────────────────────────

CATEGORIES = frozenset({
    "policy", "violation", "killswitch", "quarantine",
    "escalation", "access", "config", "alert",
})

SEVERITIES = frozenset({"info", "warning", "critical"})


@dataclass
class AuditEvent:
    """A single audit log entry."""

    seq: int
    timestamp: str  # ISO-8601
    category: str
    severity: str
    message: str
    source: str = ""
    actor: str = ""
    target: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    prev_hash: str = ""
    hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _compute_hash(event: AuditEvent) -> str:
    """SHA-256 over canonical fields (excludes the hash itself)."""
    payload = (
        f"{event.seq}|{event.timestamp}|{event.category}|{event.severity}"
        f"|{event.message}|{event.source}|{event.actor}|{event.target}"
        f"|{json.dumps(event.metadata, sort_keys=True)}|{event.prev_hash}"
    )
    return hashlib.sha256(payload.encode()).hexdigest()


# ── Audit Trail ──────────────────────────────────────────────────────

class AuditTrail:
    """Append-only, hash-chained safety event log."""

    def __init__(self) -> None:
        self._events: List[AuditEvent] = []

    # ── core operations ──────────────────────────────────────────────

    def log(
        self,
        category: str,
        severity: str,
        message: str,
        *,
        source: str = "",
        actor: str = "",
        target: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[str] = None,
    ) -> AuditEvent:
        """Append a new event to the trail."""
        if category not in CATEGORIES:
            raise ValueError(f"Unknown category: {category!r} (valid: {sorted(CATEGORIES)})")
        if severity not in SEVERITIES:
            raise ValueError(f"Unknown severity: {severity!r} (valid: {sorted(SEVERITIES)})")

        seq = len(self._events)
        prev_hash = self._events[-1].hash if self._events else "genesis"
        ts = timestamp or datetime.now(timezone.utc).isoformat()

        event = AuditEvent(
            seq=seq,
            timestamp=ts,
            category=category,
            severity=severity,
            message=message,
            source=source,
            actor=actor,
            target=target,
            metadata=metadata or {},
            prev_hash=prev_hash,
        )
        event.hash = _compute_hash(event)
        self._events.append(event)
        return event

    @property
    def events(self) -> List[AuditEvent]:
        return list(self._events)

    def __len__(self) -> int:
        return len(self._events)

    # ── search / filter ──────────────────────────────────────────────

    def search(
        self,
        *,
        category: Optional[str] = None,
        severity: Optional[str] = None,
        keyword: Optional[str] = None,
        source: Optional[str] = None,
        actor: Optional[str] = None,
        after: Optional[str] = None,
        before: Optional[str] = None,
        limit: int = 0,
    ) -> List[AuditEvent]:
        """Filter events by criteria."""
        results: List[AuditEvent] = []
        for ev in self._events:
            if category and ev.category != category:
                continue
            if severity and ev.severity != severity:
                continue
            if source and source.lower() not in ev.source.lower():
                continue
            if actor and actor.lower() not in ev.actor.lower():
                continue
            if keyword and keyword.lower() not in ev.message.lower():
                continue
            if after and ev.timestamp < after:
                continue
            if before and ev.timestamp > before:
                continue
            results.append(ev)
            if limit and len(results) >= limit:
                break
        return results

    # ── integrity verification ───────────────────────────────────────

    def verify(self) -> List[Dict[str, Any]]:
        """Verify hash chain integrity. Returns list of issues (empty = OK)."""
        issues: List[Dict[str, Any]] = []
        for i, ev in enumerate(self._events):
            expected_prev = self._events[i - 1].hash if i > 0 else "genesis"
            if ev.prev_hash != expected_prev:
                issues.append({
                    "seq": ev.seq,
                    "issue": "prev_hash mismatch",
                    "expected": expected_prev,
                    "actual": ev.prev_hash,
                })
            recomputed = _compute_hash(ev)
            if ev.hash != recomputed:
                issues.append({
                    "seq": ev.seq,
                    "issue": "hash mismatch (event tampered)",
                    "expected": recomputed,
                    "actual": ev.hash,
                })
            if ev.seq != i:
                issues.append({
                    "seq": ev.seq,
                    "issue": "sequence gap",
                    "expected": i,
                    "actual": ev.seq,
                })
        return issues

    # ── statistics ───────────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        """Compute summary statistics."""
        by_cat: Dict[str, int] = {}
        by_sev: Dict[str, int] = {}
        by_source: Dict[str, int] = {}
        for ev in self._events:
            by_cat[ev.category] = by_cat.get(ev.category, 0) + 1
            by_sev[ev.severity] = by_sev.get(ev.severity, 0) + 1
            if ev.source:
                by_source[ev.source] = by_source.get(ev.source, 0) + 1
        return {
            "total_events": len(self._events),
            "by_category": by_cat,
            "by_severity": by_sev,
            "by_source": by_source,
            "chain_intact": len(self.verify()) == 0,
            "first_event": self._events[0].timestamp if self._events else None,
            "last_event": self._events[-1].timestamp if self._events else None,
        }

    # ── export ───────────────────────────────────────────────────────

    def export_json(self, events: Optional[List[AuditEvent]] = None) -> str:
        """Export events as JSON."""
        evs = events if events is not None else self._events
        return json.dumps([e.to_dict() for e in evs], indent=2)

    def export_csv(self, events: Optional[List[AuditEvent]] = None) -> str:
        """Export events as CSV."""
        evs = events if events is not None else self._events
        buf = io.StringIO()
        fields = ["seq", "timestamp", "category", "severity", "message",
                   "source", "actor", "target", "hash"]
        writer = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for ev in evs:
            writer.writerow(ev.to_dict())
        return buf.getvalue()

    def export_html(self, events: Optional[List[AuditEvent]] = None) -> str:
        """Export events as a self-contained HTML timeline."""
        evs = events if events is not None else self._events
        integrity = self.verify()
        s = self.stats() if self._events else {}

        sev_colors = {"info": "#3b82f6", "warning": "#f59e0b", "critical": "#ef4444"}
        cat_icons = {
            "policy": "📋", "violation": "🚨", "killswitch": "🔴",
            "quarantine": "🔒", "escalation": "⬆️", "access": "🔑",
            "config": "⚙️", "alert": "🔔",
        }

        rows = []
        for ev in evs:
            icon = cat_icons.get(ev.category, "•")
            color = sev_colors.get(ev.severity, "#666")
            meta = _html.escape(json.dumps(ev.metadata)) if ev.metadata else ""
            rows.append(f"""
            <div class="event" style="border-left:4px solid {color}">
              <div class="header">
                <span class="icon">{icon}</span>
                <span class="cat">{_html.escape(ev.category)}</span>
                <span class="sev" style="color:{color}">{_html.escape(ev.severity.upper())}</span>
                <span class="ts">{_html.escape(ev.timestamp)}</span>
                <span class="seq">#{ev.seq}</span>
              </div>
              <div class="msg">{_html.escape(ev.message)}</div>
              <div class="details">
                {f'<span>Source: {_html.escape(ev.source)}</span>' if ev.source else ''}
                {f'<span>Actor: {_html.escape(ev.actor)}</span>' if ev.actor else ''}
                {f'<span>Target: {_html.escape(ev.target)}</span>' if ev.target else ''}
              </div>
              <div class="hash" title="{_html.escape(ev.hash)}">{_html.escape(ev.hash[:16])}…</div>
              {f'<div class="meta">{meta}</div>' if meta else ''}
            </div>""")

        chain_status = "✅ Chain Intact" if not integrity else f"⚠️ {len(integrity)} issue(s)"
        total = s.get("total_events", 0)

        return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Safety Audit Trail</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:system-ui,sans-serif;background:#0f172a;color:#e2e8f0;padding:20px;max-width:900px;margin:0 auto}}
h1{{font-size:1.5rem;margin-bottom:4px}}
.summary{{display:flex;gap:16px;flex-wrap:wrap;margin:12px 0 20px}}
.stat{{background:#1e293b;padding:8px 14px;border-radius:8px;font-size:.85rem}}
.stat b{{color:#38bdf8}}
.filter-bar{{margin-bottom:16px}}
.filter-bar input,.filter-bar select{{background:#1e293b;border:1px solid #334155;color:#e2e8f0;padding:6px 10px;border-radius:6px;margin-right:8px;font-size:.85rem}}
.event{{background:#1e293b;border-radius:8px;padding:12px 16px;margin-bottom:8px}}
.header{{display:flex;align-items:center;gap:8px;flex-wrap:wrap;font-size:.85rem}}
.icon{{font-size:1.1rem}}
.cat{{background:#334155;padding:2px 8px;border-radius:4px}}
.sev{{font-weight:700;text-transform:uppercase;font-size:.75rem}}
.ts{{color:#94a3b8;margin-left:auto}}
.seq{{color:#64748b;font-size:.75rem}}
.msg{{margin:6px 0;font-size:.95rem}}
.details{{display:flex;gap:12px;font-size:.8rem;color:#94a3b8}}
.hash{{font-family:monospace;font-size:.7rem;color:#475569;margin-top:4px}}
.meta{{font-family:monospace;font-size:.75rem;color:#64748b;margin-top:4px}}
</style></head><body>
<h1>🔗 Safety Audit Trail</h1>
<div class="summary">
  <div class="stat"><b>{total}</b> events</div>
  <div class="stat">{chain_status}</div>
</div>
<div class="filter-bar">
  <input type="text" id="q" placeholder="Search…" oninput="filterEvents()">
  <select id="catF" onchange="filterEvents()"><option value="">All categories</option>
    {''.join(f'<option>{c}</option>' for c in sorted(CATEGORIES))}
  </select>
  <select id="sevF" onchange="filterEvents()"><option value="">All severities</option>
    {''.join(f'<option>{s}</option>' for s in sorted(SEVERITIES))}
  </select>
</div>
<div id="events">{''.join(rows)}</div>
<script>
function filterEvents(){{
  const q=document.getElementById('q').value.toLowerCase();
  const c=document.getElementById('catF').value;
  const s=document.getElementById('sevF').value;
  document.querySelectorAll('.event').forEach(e=>{{
    const cat=e.querySelector('.cat').textContent;
    const sev=e.querySelector('.sev').textContent.toLowerCase();
    const txt=e.textContent.toLowerCase();
    e.style.display=(!q||txt.includes(q))&&(!c||cat===c)&&(!s||sev===s)?'':'none';
  }});
}}
</script></body></html>"""

    # ── serialization ────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return {"events": [e.to_dict() for e in self._events]}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AuditTrail":
        trail = cls()
        for d in data.get("events", []):
            ev = AuditEvent(**{k: v for k, v in d.items()})
            trail._events.append(ev)
        return trail

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "AuditTrail":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


# ── CLI ──────────────────────────────────────────────────────────────

def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="replication audit-trail",
        description="Safety Audit Trail — tamper-evident event log with hash chaining",
    )
    sub = parser.add_subparsers(dest="cmd")

    # log
    p_log = sub.add_parser("log", help="Record a new audit event")
    p_log.add_argument("--category", "-c", required=True, choices=sorted(CATEGORIES))
    p_log.add_argument("--severity", "-s", required=True, choices=sorted(SEVERITIES))
    p_log.add_argument("--message", "-m", required=True)
    p_log.add_argument("--source", default="")
    p_log.add_argument("--actor", default="")
    p_log.add_argument("--target", default="")
    p_log.add_argument("--file", "-f", default="audit_trail.json", help="Trail file")

    # search
    p_search = sub.add_parser("search", help="Search/filter events")
    p_search.add_argument("--category", "-c")
    p_search.add_argument("--severity", "-s")
    p_search.add_argument("--keyword", "-k")
    p_search.add_argument("--source")
    p_search.add_argument("--actor")
    p_search.add_argument("--after")
    p_search.add_argument("--before")
    p_search.add_argument("--limit", type=int, default=0)
    p_search.add_argument("--file", "-f", default="audit_trail.json")

    # verify
    p_verify = sub.add_parser("verify", help="Verify chain integrity")
    p_verify.add_argument("--file", "-f", default="audit_trail.json")

    # export
    p_export = sub.add_parser("export", help="Export trail")
    p_export.add_argument("--format", choices=["json", "csv", "html"], default="json")
    p_export.add_argument("--output", "-o")
    p_export.add_argument("--file", "-f", default="audit_trail.json")

    # stats
    p_stats = sub.add_parser("stats", help="Show summary statistics")
    p_stats.add_argument("--file", "-f", default="audit_trail.json")

    args = parser.parse_args(argv)

    if not args.cmd:
        parser.print_help()
        return

    # Load or create trail
    trail_file = args.file
    try:
        trail = AuditTrail.load(trail_file)
    except FileNotFoundError:
        trail = AuditTrail()

    if args.cmd == "log":
        ev = trail.log(
            category=args.category,
            severity=args.severity,
            message=args.message,
            source=args.source,
            actor=args.actor,
            target=args.target,
        )
        trail.save(trail_file)
        print(f"Logged event #{ev.seq} [{ev.category}/{ev.severity}]: {ev.message}")
        print(f"Hash: {ev.hash}")

    elif args.cmd == "search":
        results = trail.search(
            category=args.category,
            severity=args.severity,
            keyword=args.keyword,
            source=args.source,
            actor=args.actor,
            after=args.after,
            before=args.before,
            limit=args.limit,
        )
        print(f"Found {len(results)} event(s):")
        for ev in results:
            print(f"  #{ev.seq} [{ev.category}/{ev.severity}] {ev.timestamp} — {ev.message}")

    elif args.cmd == "verify":
        issues = trail.verify()
        if not issues:
            print(f"✅ Chain integrity verified — {len(trail)} events, no tampering detected.")
        else:
            print(f"⚠️ {len(issues)} integrity issue(s) found:")
            for iss in issues:
                print(f"  Event #{iss['seq']}: {iss['issue']}")

    elif args.cmd == "export":
        fmt = args.format
        if fmt == "json":
            out = trail.export_json()
        elif fmt == "csv":
            out = trail.export_csv()
        else:
            out = trail.export_html()

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(out)
            print(f"Exported {len(trail)} events to {args.output} ({fmt})")
        else:
            print(out)

    elif args.cmd == "stats":
        if not trail.events:
            print("No events recorded.")
            return
        st = trail.stats()
        print(f"Total events: {st['total_events']}")
        print(f"Chain intact: {'Yes' if st['chain_intact'] else 'NO'}")
        print(f"Time range: {st['first_event']} → {st['last_event']}")
        print("By category:")
        for k, v in sorted(st["by_category"].items()):
            print(f"  {k}: {v}")
        print("By severity:")
        for k, v in sorted(st["by_severity"].items()):
            print(f"  {k}: {v}")
        if st["by_source"]:
            print("By source:")
            for k, v in sorted(st["by_source"].items()):
                print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
