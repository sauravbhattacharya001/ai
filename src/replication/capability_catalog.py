"""Capability Catalog — track, classify, and manage observed agent capabilities.

Maintains a catalog of agent capabilities with risk tiers, approval status,
first-seen dates, and notes.  Useful for governance teams that need to know
what an agent population can do and whether each capability has been reviewed.

CLI usage::

    python -m replication capability-catalog --list
    python -m replication capability-catalog --add "code execution" --risk high
    python -m replication capability-catalog --approve "code execution"
    python -m replication capability-catalog --revoke "code execution"
    python -m replication capability-catalog --search network
    python -m replication capability-catalog --export catalog.json
    python -m replication capability-catalog --stats
    python -m replication capability-catalog --audit
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


# ── Risk tiers ───────────────────────────────────────────────────────

RISK_TIERS = ("low", "medium", "high", "critical")

RISK_COLORS = {
    "low": "\033[32m",       # green
    "medium": "\033[33m",    # yellow
    "high": "\033[31m",      # red
    "critical": "\033[35m",  # magenta
}
RESET = "\033[0m"


# ── Catalog entry ────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _new_entry(
    name: str,
    risk: str = "medium",
    description: str = "",
    tags: Optional[List[str]] = None,
) -> Dict[str, Any]:
    return {
        "name": name,
        "risk": risk,
        "status": "pending",
        "description": description,
        "tags": tags or [],
        "first_seen": _now_iso(),
        "last_updated": _now_iso(),
        "approved_by": None,
        "approved_at": None,
        "notes": [],
    }


# ── Catalog store ────────────────────────────────────────────────────

DEFAULT_PATH = os.path.join(os.getcwd(), "capability_catalog.json")


def _load(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def _save(entries: List[Dict[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)


def _find(entries: List[Dict[str, Any]], name: str) -> Optional[Dict[str, Any]]:
    key = name.lower().strip()
    for e in entries:
        if e["name"].lower().strip() == key:
            return e
    return None


# ── Actions ──────────────────────────────────────────────────────────

def list_capabilities(entries: List[Dict[str, Any]], risk_filter: Optional[str] = None, status_filter: Optional[str] = None) -> None:
    filtered = entries
    if risk_filter:
        filtered = [e for e in filtered if e["risk"] == risk_filter]
    if status_filter:
        filtered = [e for e in filtered if e["status"] == status_filter]

    if not filtered:
        print("No capabilities found.")
        return

    # Header
    print(f"\n{'Capability':<30} {'Risk':<10} {'Status':<12} {'First Seen':<22} {'Tags'}")
    print("─" * 95)

    for e in sorted(filtered, key=lambda x: RISK_TIERS.index(x.get("risk", "medium"))):
        risk = e.get("risk", "medium")
        color = RISK_COLORS.get(risk, "")
        status = e.get("status", "pending")
        tags = ", ".join(e.get("tags", []))
        first = e.get("first_seen", "")[:10]
        print(f"{e['name']:<30} {color}{risk:<10}{RESET} {status:<12} {first:<22} {tags}")

    print(f"\nTotal: {len(filtered)} capabilities\n")


def add_capability(
    entries: List[Dict[str, Any]],
    name: str,
    risk: str,
    description: str,
    tags: List[str],
    path: str,
) -> None:
    if _find(entries, name):
        print(f"⚠  Capability '{name}' already exists. Use --update to modify.")
        return
    if risk not in RISK_TIERS:
        print(f"Invalid risk tier '{risk}'. Choose from: {', '.join(RISK_TIERS)}")
        return
    entry = _new_entry(name, risk, description, tags)
    entries.append(entry)
    _save(entries, path)
    color = RISK_COLORS.get(risk, "")
    print(f"✅ Added '{name}' ({color}{risk}{RESET}) to catalog.")


def approve_capability(entries: List[Dict[str, Any]], name: str, approver: str, path: str) -> None:
    entry = _find(entries, name)
    if not entry:
        print(f"❌ Capability '{name}' not found.")
        return
    entry["status"] = "approved"
    entry["approved_by"] = approver
    entry["approved_at"] = _now_iso()
    entry["last_updated"] = _now_iso()
    _save(entries, path)
    print(f"✅ '{name}' approved by {approver}.")


def revoke_capability(entries: List[Dict[str, Any]], name: str, reason: str, path: str) -> None:
    entry = _find(entries, name)
    if not entry:
        print(f"❌ Capability '{name}' not found.")
        return
    entry["status"] = "revoked"
    entry["last_updated"] = _now_iso()
    entry["notes"].append({"action": "revoked", "reason": reason, "at": _now_iso()})
    _save(entries, path)
    print(f"🚫 '{name}' revoked. Reason: {reason or 'none given'}")


def search_capabilities(entries: List[Dict[str, Any]], query: str) -> None:
    q = query.lower()
    matches = [
        e for e in entries
        if q in e["name"].lower()
        or q in e.get("description", "").lower()
        or any(q in t.lower() for t in e.get("tags", []))
    ]
    if not matches:
        print(f"No capabilities matching '{query}'.")
        return
    list_capabilities(matches)


def show_stats(entries: List[Dict[str, Any]]) -> None:
    if not entries:
        print("Catalog is empty.")
        return

    by_risk = {t: 0 for t in RISK_TIERS}
    by_status = {"pending": 0, "approved": 0, "revoked": 0}
    for e in entries:
        by_risk[e.get("risk", "medium")] += 1
        s = e.get("status", "pending")
        by_status[s] = by_status.get(s, 0) + 1

    print("\n📊 Capability Catalog Stats")
    print("─" * 40)
    print(f"Total capabilities: {len(entries)}\n")

    print("By risk tier:")
    for tier in RISK_TIERS:
        color = RISK_COLORS.get(tier, "")
        bar = "█" * by_risk[tier]
        print(f"  {color}{tier:<10}{RESET} {by_risk[tier]:>3}  {bar}")

    print("\nBy status:")
    for status, count in by_status.items():
        icon = {"pending": "⏳", "approved": "✅", "revoked": "🚫"}.get(status, "•")
        print(f"  {icon} {status:<10} {count:>3}")

    # Pending high/critical — flag for review
    urgent = [e for e in entries if e["status"] == "pending" and e["risk"] in ("high", "critical")]
    if urgent:
        print(f"\n⚠  {len(urgent)} high/critical capabilities pending review:")
        for e in urgent:
            color = RISK_COLORS.get(e["risk"], "")
            print(f"    {color}•{RESET} {e['name']} ({e['risk']})")
    print()


def audit_catalog(entries: List[Dict[str, Any]]) -> None:
    """Check catalog health — stale entries, missing fields, etc."""
    issues: List[str] = []

    for e in entries:
        name = e.get("name", "<unnamed>")
        if not e.get("description"):
            issues.append(f"  ⚠  '{name}' has no description")
        if not e.get("tags"):
            issues.append(f"  ⚠  '{name}' has no tags")
        if e.get("status") == "approved" and not e.get("approved_by"):
            issues.append(f"  ❌ '{name}' approved but no approver recorded")
        if e.get("risk") == "critical" and e.get("status") == "pending":
            issues.append(f"  🔴 '{name}' is CRITICAL and still pending review")

    print("\n🔍 Catalog Audit")
    print("─" * 40)
    if issues:
        print(f"Found {len(issues)} issue(s):\n")
        for issue in issues:
            print(issue)
    else:
        print("✅ No issues found — catalog looks healthy.")
    print()


def export_catalog(entries: List[Dict[str, Any]], out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)
    print(f"📁 Exported {len(entries)} capabilities to {out_path}")


# ── CLI ──────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m replication capability-catalog",
        description="Capability Catalog — track and manage observed agent capabilities",
    )
    parser.add_argument("--catalog", default=DEFAULT_PATH, help="Path to catalog JSON file")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true", help="List all capabilities")
    group.add_argument("--add", metavar="NAME", help="Add a new capability")
    group.add_argument("--approve", metavar="NAME", help="Approve a capability")
    group.add_argument("--revoke", metavar="NAME", help="Revoke a capability")
    group.add_argument("--search", metavar="QUERY", help="Search capabilities")
    group.add_argument("--stats", action="store_true", help="Show catalog statistics")
    group.add_argument("--audit", action="store_true", help="Audit catalog for issues")
    group.add_argument("--export", metavar="PATH", help="Export catalog to JSON")

    # Options for --add
    parser.add_argument("--risk", default="medium", choices=RISK_TIERS, help="Risk tier")
    parser.add_argument("--description", default="", help="Capability description")
    parser.add_argument("--tags", default="", help="Comma-separated tags")

    # Options for --approve
    parser.add_argument("--approver", default="system", help="Who is approving")

    # Options for --revoke
    parser.add_argument("--reason", default="", help="Revocation reason")

    # Filters for --list
    parser.add_argument("--risk-filter", choices=RISK_TIERS, help="Filter by risk tier")
    parser.add_argument("--status-filter", choices=("pending", "approved", "revoked"), help="Filter by status")

    args = parser.parse_args(argv)
    entries = _load(args.catalog)

    if args.list:
        list_capabilities(entries, args.risk_filter, args.status_filter)
    elif args.add:
        tags = [t.strip() for t in args.tags.split(",") if t.strip()] if args.tags else []
        add_capability(entries, args.add, args.risk, args.description, tags, args.catalog)
    elif args.approve:
        approve_capability(entries, args.approve, args.approver, args.catalog)
    elif args.revoke:
        revoke_capability(entries, args.revoke, args.reason, args.catalog)
    elif args.search:
        search_capabilities(entries, args.search)
    elif args.stats:
        show_stats(entries)
    elif args.audit:
        audit_catalog(entries)
    elif args.export:
        export_catalog(entries, args.export)


if __name__ == "__main__":
    main()
