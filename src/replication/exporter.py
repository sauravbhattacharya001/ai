"""Audit Trail Exporter — structured data export for simulation analysis.

Exports simulation results into portable, machine-readable formats for
external analysis in pandas, R, Excel, or database tools.  Supports CSV,
JSONL (JSON Lines), and plain JSON.

Usage (CLI)::

    python -m replication.exporter                          # export default simulation
    python -m replication.exporter --format csv             # CSV tables
    python -m replication.exporter --format jsonl           # JSON Lines
    python -m replication.exporter --format json            # single JSON file
    python -m replication.exporter --format all             # all formats
    python -m replication.exporter -o ./exports             # output directory
    python -m replication.exporter --strategy greedy        # specific strategy
    python -m replication.exporter --preset stress          # built-in preset
    python -m replication.exporter --include workers timeline audit summary
    python -m replication.exporter --seed 42                # reproducible export

Programmatic::

    from replication.exporter import AuditExporter, ExportConfig
    from replication.simulator import Simulator

    sim = Simulator()
    report = sim.run()

    exporter = AuditExporter()

    # Export to CSV strings
    csv_data = exporter.workers_csv(report)
    timeline_csv = exporter.timeline_csv(report)

    # Export everything to a directory
    exporter.export_all(report, output_dir="./exports", formats=["csv", "jsonl"])

    # Get JSONL strings
    jsonl = exporter.timeline_jsonl(report)

    # Summary statistics as dict
    summary = exporter.summary_stats(report)
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

from .simulator import PRESETS, ScenarioConfig, SimulationReport, Simulator


@dataclass
class ExportConfig:
    """Configuration for audit trail exports."""

    formats: List[str] = field(default_factory=lambda: ["csv"])
    """Output formats: 'csv', 'jsonl', 'json', or 'all'."""

    include: List[str] = field(
        default_factory=lambda: ["workers", "timeline", "audit", "summary"]
    )
    """Data sections to export: 'workers', 'timeline', 'audit', 'summary'."""

    output_dir: Optional[str] = None
    """Directory to write files.  ``None`` means return strings only."""

    timestamp_format: str = "iso"
    """Timestamp format: 'iso', 'epoch', or 'human'."""


@dataclass
class ExportResult:
    """Result of an export operation."""

    files_written: List[str] = field(default_factory=list)
    """Paths of files created on disk."""

    data: Dict[str, str] = field(default_factory=dict)
    """In-memory exported data keyed by ``section.format``."""

    summary: Dict[str, Any] = field(default_factory=dict)
    """Computed summary statistics."""

    def render(self) -> str:
        """Human-readable summary of the export."""
        lines: List[str] = []
        lines.append("=== Audit Trail Export ===")
        lines.append("")
        if self.files_written:
            lines.append(f"Files written: {len(self.files_written)}")
            for f in self.files_written:
                lines.append(f"  {f}")
        else:
            lines.append("Files written: 0 (in-memory only)")
        lines.append("")
        if self.summary:
            lines.append("Summary Statistics:")
            for key, val in self.summary.items():
                lines.append(f"  {key}: {val}")
        return "\n".join(lines)


class AuditExporter:
    """Exports simulation audit trails to structured formats."""

    def __init__(self, config: Optional[ExportConfig] = None) -> None:
        self.config = config or ExportConfig()

    # ── Worker Lifecycle ─────────────────────────────────────

    def workers_csv(self, report: SimulationReport) -> str:
        """Export worker lifecycle data as CSV."""
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "worker_id",
            "parent_id",
            "depth",
            "tasks_completed",
            "replications_attempted",
            "replications_succeeded",
            "replications_denied",
            "children_count",
            "is_leaf",
            "replication_rate",
            "denial_rate",
        ])

        for wid, rec in sorted(report.workers.items()):
            repl_rate = (
                round(rec.replications_succeeded / rec.replications_attempted, 4)
                if rec.replications_attempted > 0
                else 0.0
            )
            denial_rate = (
                round(rec.replications_denied / rec.replications_attempted, 4)
                if rec.replications_attempted > 0
                else 0.0
            )
            writer.writerow([
                wid,
                rec.parent_id or "",
                rec.depth,
                rec.tasks_completed,
                rec.replications_attempted,
                rec.replications_succeeded,
                rec.replications_denied,
                len(rec.children),
                1 if len(rec.children) == 0 else 0,
                repl_rate,
                denial_rate,
            ])

        return output.getvalue()

    def workers_jsonl(self, report: SimulationReport) -> str:
        """Export worker lifecycle data as JSON Lines."""
        lines: List[str] = []
        for wid, rec in sorted(report.workers.items()):
            repl_rate = (
                round(rec.replications_succeeded / rec.replications_attempted, 4)
                if rec.replications_attempted > 0
                else 0.0
            )
            entry = {
                "worker_id": wid,
                "parent_id": rec.parent_id,
                "depth": rec.depth,
                "tasks_completed": rec.tasks_completed,
                "replications_attempted": rec.replications_attempted,
                "replications_succeeded": rec.replications_succeeded,
                "replications_denied": rec.replications_denied,
                "children": rec.children,
                "children_count": len(rec.children),
                "is_leaf": len(rec.children) == 0,
                "replication_rate": repl_rate,
            }
            lines.append(json.dumps(entry, separators=(",", ":")))
        return "\n".join(lines) + "\n" if lines else ""

    # ── Timeline Events ──────────────────────────────────────

    def timeline_csv(self, report: SimulationReport) -> str:
        """Export timeline events as CSV."""
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "sequence",
            "timestamp_ms",
            "event_type",
            "worker_id",
            "details",
        ])

        for idx, event in enumerate(report.timeline):
            event_type = event.get("event", event.get("type", "unknown"))
            worker_id = event.get("worker", event.get("worker_id", ""))
            ts = event.get("elapsed_ms", event.get("timestamp_ms", 0.0))

            # Flatten remaining keys as details
            skip_keys = {"event", "type", "worker", "worker_id", "elapsed_ms",
                         "timestamp_ms"}
            details = {k: v for k, v in event.items() if k not in skip_keys}
            writer.writerow([
                idx,
                round(ts, 2),
                event_type,
                worker_id,
                json.dumps(details, separators=(",", ":")) if details else "",
            ])

        return output.getvalue()

    def timeline_jsonl(self, report: SimulationReport) -> str:
        """Export timeline events as JSON Lines."""
        lines: List[str] = []
        for idx, event in enumerate(report.timeline):
            entry = {"sequence": idx, **event}
            lines.append(json.dumps(entry, separators=(",", ":")))
        return "\n".join(lines) + "\n" if lines else ""

    # ── Audit Events ─────────────────────────────────────────

    def audit_csv(self, report: SimulationReport) -> str:
        """Export raw audit events as CSV."""
        output = io.StringIO()
        writer = csv.writer(output)

        if not report.audit_events:
            writer.writerow(["(no audit events)"])
            return output.getvalue()

        # Use keys from first event as header
        all_keys: List[str] = []
        seen: set = set()
        for evt in report.audit_events:
            for k in evt:
                if k not in seen:
                    all_keys.append(k)
                    seen.add(k)

        writer.writerow(all_keys)
        for evt in report.audit_events:
            row = []
            for k in all_keys:
                val = evt.get(k, "")
                if isinstance(val, (dict, list)):
                    row.append(json.dumps(val, separators=(",", ":"), default=str))
                elif isinstance(val, datetime):
                    row.append(val.isoformat())
                else:
                    row.append(str(val) if val is not None else "")
            writer.writerow(row)

        return output.getvalue()

    def audit_jsonl(self, report: SimulationReport) -> str:
        """Export audit events as JSON Lines."""
        lines: List[str] = []
        for evt in report.audit_events:
            lines.append(json.dumps(evt, separators=(",", ":"), default=str))
        return "\n".join(lines) + "\n" if lines else ""

    # ── Summary Statistics ───────────────────────────────────

    def summary_stats(self, report: SimulationReport) -> Dict[str, Any]:
        """Compute comprehensive summary statistics from the report."""
        workers = report.workers
        total_workers = len(workers)
        leaf_count = sum(
            1 for w in workers.values() if len(w.children) == 0
        )
        depths = [w.depth for w in workers.values()]
        max_depth_reached = max(depths, default=0)
        avg_depth = round(sum(depths) / total_workers, 2) if total_workers else 0.0

        tasks_per_worker = [w.tasks_completed for w in workers.values()]
        avg_tasks = (
            round(sum(tasks_per_worker) / total_workers, 2)
            if total_workers
            else 0.0
        )

        # Children distribution
        children_counts = [len(w.children) for w in workers.values()]
        max_children = max(children_counts, default=0)
        avg_children = (
            round(sum(children_counts) / total_workers, 2)
            if total_workers
            else 0.0
        )

        # Replication rates
        total_attempted = report.total_replications_attempted
        total_succeeded = report.total_replications_succeeded
        total_denied = report.total_replications_denied
        success_rate = (
            round(total_succeeded / total_attempted, 4)
            if total_attempted > 0
            else 0.0
        )
        denial_rate = (
            round(total_denied / total_attempted, 4)
            if total_attempted > 0
            else 0.0
        )

        # Workers per depth level
        depth_distribution: Dict[int, int] = {}
        for d in depths:
            depth_distribution[d] = depth_distribution.get(d, 0) + 1

        return {
            "strategy": report.config.strategy,
            "max_depth_config": report.config.max_depth,
            "max_replicas_config": report.config.max_replicas,
            "total_workers": total_workers,
            "leaf_workers": leaf_count,
            "internal_workers": total_workers - leaf_count,
            "max_depth_reached": max_depth_reached,
            "avg_depth": avg_depth,
            "total_tasks": report.total_tasks,
            "avg_tasks_per_worker": avg_tasks,
            "replications_attempted": total_attempted,
            "replications_succeeded": total_succeeded,
            "replications_denied": total_denied,
            "success_rate": success_rate,
            "denial_rate": denial_rate,
            "max_children": max_children,
            "avg_children": avg_children,
            "depth_distribution": depth_distribution,
            "duration_ms": round(report.duration_ms, 2),
            "timeline_events": len(report.timeline),
            "audit_events": len(report.audit_events),
        }

    def summary_csv(self, report: SimulationReport) -> str:
        """Export summary statistics as CSV (key-value pairs)."""
        stats = self.summary_stats(report)
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["metric", "value"])
        for key, val in stats.items():
            if isinstance(val, dict):
                writer.writerow([key, json.dumps(val, separators=(",", ":"))])
            else:
                writer.writerow([key, val])
        return output.getvalue()

    def summary_json(self, report: SimulationReport) -> str:
        """Export summary statistics as formatted JSON."""
        stats = self.summary_stats(report)
        return json.dumps(stats, indent=2) + "\n"

    # ── Combined Export ──────────────────────────────────────

    def export_all(
        self,
        report: SimulationReport,
        output_dir: Optional[str] = None,
        formats: Optional[List[str]] = None,
        include: Optional[List[str]] = None,
    ) -> ExportResult:
        """Export all requested sections in all requested formats.

        Parameters
        ----------
        report : SimulationReport
            The simulation report to export.
        output_dir : str, optional
            Directory to write files.  Created if it doesn't exist.
            If ``None``, data is returned in-memory only.
        formats : list of str, optional
            Formats to export: ``'csv'``, ``'jsonl'``, ``'json'``, or
            ``'all'``.  Defaults to config value.
        include : list of str, optional
            Sections: ``'workers'``, ``'timeline'``, ``'audit'``,
            ``'summary'``.  Defaults to config value.

        Returns
        -------
        ExportResult
            Contains file paths written and/or in-memory data.
        """
        fmts = formats or self.config.formats
        if "all" in fmts:
            fmts = ["csv", "jsonl", "json"]
        sections = include or self.config.include

        result = ExportResult(summary=self.summary_stats(report))

        # Section → format → (method, extension)
        exporters: Dict[str, Dict[str, tuple]] = {
            "workers": {
                "csv": (self.workers_csv, "csv"),
                "jsonl": (self.workers_jsonl, "jsonl"),
            },
            "timeline": {
                "csv": (self.timeline_csv, "csv"),
                "jsonl": (self.timeline_jsonl, "jsonl"),
            },
            "audit": {
                "csv": (self.audit_csv, "csv"),
                "jsonl": (self.audit_jsonl, "jsonl"),
            },
            "summary": {
                "csv": (self.summary_csv, "csv"),
                "json": (self.summary_json, "json"),
                "jsonl": (self.summary_json, "json"),  # summary is always JSON
            },
        }

        for section in sections:
            if section not in exporters:
                continue
            for fmt in fmts:
                if fmt not in exporters[section]:
                    continue
                method, ext = exporters[section][fmt]
                data = method(report)
                key = f"{section}.{ext}"
                result.data[key] = data

                if output_dir is not None:
                    os.makedirs(output_dir, exist_ok=True)
                    filename = f"{section}.{ext}"
                    filepath = os.path.join(output_dir, filename)
                    with open(filepath, "w", encoding="utf-8", newline="") as f:
                        f.write(data)
                    result.files_written.append(filepath)

        return result

    # ── Multi-Report Export ──────────────────────────────────

    def comparative_csv(self, reports: List[SimulationReport]) -> str:
        """Export summary stats from multiple reports as a single CSV table.

        Each row represents one simulation run — useful for comparing
        strategies, presets, or parameter sweeps in a spreadsheet.
        """
        if not reports:
            return ""

        rows: List[Dict[str, Any]] = []
        for idx, report in enumerate(reports):
            stats = self.summary_stats(report)
            stats["run_index"] = idx
            # Move depth_distribution to a JSON string for CSV
            if "depth_distribution" in stats:
                stats["depth_distribution"] = json.dumps(
                    stats["depth_distribution"], separators=(",", ":")
                )
            rows.append(stats)

        output = io.StringIO()
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        return output.getvalue()


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entry point for audit trail export."""
    parser = argparse.ArgumentParser(
        description="Export simulation audit trails to CSV/JSONL/JSON",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--format", "-f",
        choices=["csv", "jsonl", "json", "all"],
        default="csv",
        help="Output format (default: csv)",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output directory (default: print to stdout)",
    )
    parser.add_argument(
        "--include",
        nargs="+",
        choices=["workers", "timeline", "audit", "summary"],
        default=["workers", "timeline", "audit", "summary"],
        help="Sections to export (default: all)",
    )
    parser.add_argument(
        "--strategy",
        choices=[s.value for s in Simulator.__init__.__code__.co_varnames[:0]]
        or ["greedy", "conservative", "random", "chain", "burst"],
        default="greedy",
        help="Simulation strategy (default: greedy)",
    )
    parser.add_argument(
        "--preset",
        choices=list(PRESETS.keys()),
        default=None,
        help="Use a built-in scenario preset",
    )
    parser.add_argument(
        "--max-depth", type=int, default=None,
        help="Override max replication depth",
    )
    parser.add_argument(
        "--max-replicas", type=int, default=None,
        help="Override max replicas",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--compare-all", action="store_true",
        help="Run all strategies and export comparative CSV",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Shortcut for --format json",
    )

    args = parser.parse_args(argv)

    if args.json:
        args.format = "json"

    # Build scenario config
    if args.preset:
        config = ScenarioConfig(**{
            k: v for k, v in PRESETS[args.preset].__dict__.items()
        })
    else:
        config = ScenarioConfig(strategy=args.strategy)

    if args.max_depth is not None:
        config.max_depth = args.max_depth
    if args.max_replicas is not None:
        config.max_replicas = args.max_replicas
    if args.seed is not None:
        config.seed = args.seed

    exporter = AuditExporter(ExportConfig(
        formats=[args.format],
        include=args.include,
        output_dir=args.output,
    ))

    if args.compare_all:
        # Run all strategies and produce comparative table
        reports: List[SimulationReport] = []
        for strat in ["greedy", "conservative", "random", "chain", "burst"]:
            cfg = ScenarioConfig(
                strategy=strat,
                max_depth=config.max_depth,
                max_replicas=config.max_replicas,
                seed=config.seed,
            )
            sim = Simulator(cfg)
            reports.append(sim.run())

        csv_data = exporter.comparative_csv(reports)
        if args.output:
            os.makedirs(args.output, exist_ok=True)
            filepath = os.path.join(args.output, "comparison.csv")
            with open(filepath, "w", encoding="utf-8", newline="") as f:
                f.write(csv_data)
            print(f"Wrote: {filepath}")
        else:
            print(csv_data)
        return

    # Single simulation export
    sim = Simulator(config)
    report = sim.run()
    result = exporter.export_all(report, output_dir=args.output)

    if args.output:
        print(result.render())
    else:
        # Print each section to stdout
        for key, data in result.data.items():
            print(f"--- {key} ---")
            print(data)
        print()
        print(result.render())


if __name__ == "__main__":
    main()
