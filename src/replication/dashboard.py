"""dashboard — self-contained HTML report generator for simulation runs.

Produces a single-file HTML dashboard with embedded CSS/JS (no external
dependencies) for reviewing and comparing simulation results.

Usage (CLI)::

    python -m replication dashboard --strategy greedy --depth 5
    python -m replication dashboard --compare greedy random --steps 100
    python -m replication dashboard -o report.html

Usage (API)::

    from replication.dashboard import DashboardGenerator
    from replication.simulator import Simulator, ScenarioConfig

    report = Simulator(ScenarioConfig(strategy="greedy")).run()
    gen = DashboardGenerator()
    html = gen.single_report(report, title="Greedy Run")
    Path("report.html").write_text(html)
"""

from __future__ import annotations

import argparse
import html as html_lib
import json
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .simulator import ScenarioConfig, SimulationReport, Simulator, WorkerRecord


# ── Configuration ────────────────────────────────────────────────

@dataclass
class DashboardConfig:
    """Options controlling dashboard generation."""
    title: str = "Replication Safety Dashboard"
    theme: str = "light"         # "light" or "dark"
    show_timeline: bool = True
    show_tree: bool = True
    show_audit: bool = True
    max_timeline_events: int = 200
    max_audit_events: int = 100


# ── HTML Templates ───────────────────────────────────────────────

_CSS = """\
:root {
  --bg: #ffffff; --fg: #1f2937; --card-bg: #f9fafb; --border: #e5e7eb;
  --accent: #4f46e5; --accent-light: #eef2ff; --green: #059669;
  --red: #dc2626; --yellow: #d97706; --blue: #2563eb;
}
[data-theme="dark"] {
  --bg: #111827; --fg: #f9fafb; --card-bg: #1f2937; --border: #374151;
  --accent: #818cf8; --accent-light: #1e1b4b;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: system-ui, -apple-system, sans-serif; background: var(--bg);
  color: var(--fg); line-height: 1.6; padding: 2rem; max-width: 1200px; margin: 0 auto; }
h1 { font-size: 1.8rem; margin-bottom: 0.25rem; }
h2 { font-size: 1.3rem; margin: 2rem 0 1rem; border-bottom: 2px solid var(--accent);
  padding-bottom: 0.3rem; }
h3 { font-size: 1.05rem; margin: 1rem 0 0.5rem; }
.subtitle { color: #6b7280; font-size: 0.9rem; margin-bottom: 2rem; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 1rem; }
.card { background: var(--card-bg); border: 1px solid var(--border);
  border-radius: 10px; padding: 1.2rem; }
.card-label { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em;
  color: #9ca3af; margin-bottom: 0.3rem; }
.card-value { font-size: 1.6rem; font-weight: 700; }
.card-sub { font-size: 0.8rem; color: #6b7280; margin-top: 0.2rem; }
.bar-chart { display: flex; flex-direction: column; gap: 0.4rem; }
.bar-row { display: flex; align-items: center; gap: 0.5rem; }
.bar-label { min-width: 100px; font-size: 0.8rem; text-align: right; }
.bar-track { flex: 1; height: 22px; background: var(--border); border-radius: 4px; overflow: hidden; }
.bar-fill { height: 100%; border-radius: 4px; display: flex; align-items: center;
  padding-left: 6px; font-size: 0.7rem; font-weight: 600; color: #fff; }
.bar-val { min-width: 50px; font-size: 0.8rem; font-weight: 600; }
table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
th, td { padding: 0.5rem 0.75rem; text-align: left; border-bottom: 1px solid var(--border); }
th { background: var(--card-bg); font-weight: 600; position: sticky; top: 0; }
.badge { display: inline-block; padding: 0.15rem 0.5rem; border-radius: 10px;
  font-size: 0.7rem; font-weight: 600; }
.badge-green { background: #d1fae5; color: #065f46; }
.badge-red { background: #fee2e2; color: #991b1b; }
.badge-yellow { background: #fef3c7; color: #92400e; }
.badge-blue { background: #dbeafe; color: #1e40af; }
pre { background: var(--card-bg); border: 1px solid var(--border); border-radius: 8px;
  padding: 1rem; overflow-x: auto; font-size: 0.8rem; line-height: 1.5; }
.compare-grid { display: grid; gap: 1rem; }
.tab-bar { display: flex; gap: 0; border-bottom: 2px solid var(--border); margin-bottom: 1rem; }
.tab-btn { padding: 0.5rem 1.2rem; border: none; background: none; cursor: pointer;
  font-size: 0.9rem; color: #6b7280; border-bottom: 2px solid transparent; margin-bottom: -2px; }
.tab-btn.active { color: var(--accent); border-bottom-color: var(--accent); font-weight: 600; }
.tab-panel { display: none; }
.tab-panel.active { display: block; }
footer { margin-top: 3rem; padding-top: 1rem; border-top: 1px solid var(--border);
  font-size: 0.75rem; color: #9ca3af; text-align: center; }
"""

_JS = """\
document.addEventListener('DOMContentLoaded', function() {
  document.querySelectorAll('.tab-bar').forEach(function(bar) {
    bar.querySelectorAll('.tab-btn').forEach(function(btn) {
      btn.addEventListener('click', function() {
        var group = btn.dataset.group;
        var target = btn.dataset.tab;
        bar.querySelectorAll('.tab-btn').forEach(function(b) { b.classList.remove('active'); });
        btn.classList.add('active');
        document.querySelectorAll('.tab-panel[data-group="' + group + '"]').forEach(function(p) {
          p.classList.remove('active');
        });
        var panel = document.getElementById(target);
        if (panel) panel.classList.add('active');
      });
    });
  });
});
"""


# ── Generator ────────────────────────────────────────────────────

class DashboardGenerator:
    """Generates self-contained HTML dashboards from simulation reports."""

    def __init__(self, config: Optional[DashboardConfig] = None) -> None:
        self.config = config or DashboardConfig()

    # ── Single report ────────────────────────────────────────────

    def single_report(
        self, report: SimulationReport, title: Optional[str] = None
    ) -> str:
        """Generate an HTML dashboard for a single simulation run."""
        title = title or self.config.title
        parts: List[str] = []
        parts.append(self._head(title))
        parts.append(f'<h1>{_esc(title)}</h1>')
        parts.append(f'<p class="subtitle">Generated {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}'
                     f' &mdash; Strategy: <strong>{_esc(report.config.strategy)}</strong></p>')

        parts.append(self._summary_cards(report))
        parts.append(self._worker_depth_chart(report))

        if self.config.show_tree:
            parts.append('<h2>Worker Lineage</h2>')
            parts.append(f'<pre>{_esc(report.render_tree())}</pre>')

        if self.config.show_timeline and report.timeline:
            parts.append(self._timeline_table(report))

        if self.config.show_audit and report.audit_events:
            parts.append(self._audit_table(report))

        parts.append(self._config_section(report))
        parts.append(self._footer())
        parts.append('</body></html>')
        return '\n'.join(parts)

    # ── Comparative report ───────────────────────────────────────

    def compare_reports(
        self, reports: List[SimulationReport], labels: Optional[List[str]] = None
    ) -> str:
        """Generate an HTML dashboard comparing multiple simulation runs."""
        if not reports:
            return self._empty_page()

        if labels is None:
            labels = [f"Run {i+1}: {r.config.strategy}" for i, r in enumerate(reports)]

        title = self.config.title + " — Comparison"
        parts: List[str] = []
        parts.append(self._head(title))
        parts.append(f'<h1>{_esc(title)}</h1>')
        parts.append(f'<p class="subtitle">Comparing {len(reports)} simulation runs &mdash; '
                     f'Generated {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}</p>')

        # Comparison table
        parts.append('<h2>Side-by-Side Comparison</h2>')
        parts.append(self._comparison_table(reports, labels))

        # Bar chart comparison
        parts.append(self._comparison_bars(reports, labels))

        # Tabbed detail panels
        parts.append('<h2>Detailed Results</h2>')
        parts.append(self._tabbed_details(reports, labels))

        parts.append(self._footer())
        parts.append('</body></html>')
        return '\n'.join(parts)

    # ── Components ───────────────────────────────────────────────

    def _head(self, title: str) -> str:
        theme = self.config.theme
        return (
            '<!DOCTYPE html>\n'
            f'<html lang="en" data-theme="{_esc(theme)}">\n'
            f'<head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">\n'
            f'<title>{_esc(title)}</title>\n'
            f'<style>{_CSS}</style>\n'
            f'<script>{_JS}</script>\n'
            '</head>\n<body>'
        )

    def _footer(self) -> str:
        return '<footer>Generated by <strong>replication.dashboard</strong> &mdash; AI Replication Safety Sandbox</footer>'

    def _empty_page(self) -> str:
        return (self._head("No Data") +
                '<h1>No simulation data</h1><p>Run a simulation first.</p>' +
                self._footer() + '</body></html>')

    def _summary_cards(self, report: SimulationReport) -> str:
        workers = report.workers
        total = len(workers)
        max_depth = max((w.depth for w in workers.values()), default=0)
        leaves = sum(1 for w in workers.values() if not w.children)
        success_rate = (
            report.total_replications_succeeded / report.total_replications_attempted * 100
            if report.total_replications_attempted > 0 else 0
        )
        denial_rate = (
            report.total_replications_denied / report.total_replications_attempted * 100
            if report.total_replications_attempted > 0 else 0
        )

        cards = [
            ("Workers", str(total), f"Max depth: {max_depth}"),
            ("Tasks", str(report.total_tasks), f"Leaves: {leaves}"),
            ("Replications", str(report.total_replications_attempted),
             f"{report.total_replications_succeeded} succeeded, {report.total_replications_denied} denied"),
            ("Success Rate", f"{success_rate:.1f}%",
             f"Denial: {denial_rate:.1f}%"),
            ("Duration", f"{report.duration_ms:.0f}ms",
             f"Strategy: {report.config.strategy}"),
            ("Timeline Events", str(len(report.timeline)),
             f"Audit: {len(report.audit_events)} events"),
        ]
        parts = ['<div class="grid">']
        for label, value, sub in cards:
            parts.append(
                f'<div class="card">'
                f'<div class="card-label">{_esc(label)}</div>'
                f'<div class="card-value">{_esc(value)}</div>'
                f'<div class="card-sub">{_esc(sub)}</div>'
                f'</div>'
            )
        parts.append('</div>')
        return '\n'.join(parts)

    def _worker_depth_chart(self, report: SimulationReport) -> str:
        depth_counts: Dict[int, int] = {}
        for w in report.workers.values():
            depth_counts[w.depth] = depth_counts.get(w.depth, 0) + 1
        if not depth_counts:
            return ''

        max_count = max(depth_counts.values())
        parts = ['<h2>Worker Distribution by Depth</h2>', '<div class="bar-chart">']
        for depth in sorted(depth_counts):
            count = depth_counts[depth]
            pct = count / max_count * 100 if max_count > 0 else 0
            color = _depth_color(depth)
            parts.append(
                f'<div class="bar-row">'
                f'<span class="bar-label">Depth {depth}</span>'
                f'<div class="bar-track"><div class="bar-fill" style="width:{pct:.0f}%;background:{color}">'
                f'{count}</div></div>'
                f'<span class="bar-val">{count}</span>'
                f'</div>'
            )
        parts.append('</div>')
        return '\n'.join(parts)

    def _timeline_table(self, report: SimulationReport) -> str:
        events = report.timeline[:self.config.max_timeline_events]
        parts = ['<h2>Timeline Events</h2>',
                 '<div style="max-height:400px;overflow-y:auto">',
                 '<table><thead><tr><th>Step</th><th>Event</th><th>Worker</th><th>Details</th></tr></thead><tbody>']
        for ev in events:
            step = ev.get('step', '')
            event_type = ev.get('event', ev.get('type', ''))
            worker = ev.get('worker_id', ev.get('worker', ''))
            details = ev.get('details', ev.get('reason', ''))
            badge_cls = _event_badge(str(event_type))
            parts.append(
                f'<tr><td>{_esc(str(step))}</td>'
                f'<td><span class="badge {badge_cls}">{_esc(str(event_type))}</span></td>'
                f'<td>{_esc(str(worker))}</td>'
                f'<td>{_esc(str(details))}</td></tr>'
            )
        parts.append('</tbody></table></div>')
        if len(report.timeline) > self.config.max_timeline_events:
            parts.append(f'<p class="card-sub">Showing {self.config.max_timeline_events} of {len(report.timeline)} events</p>')
        return '\n'.join(parts)

    def _audit_table(self, report: SimulationReport) -> str:
        events = report.audit_events[:self.config.max_audit_events]
        parts = ['<h2>Audit Log</h2>',
                 '<div style="max-height:400px;overflow-y:auto">',
                 '<table><thead><tr><th>Step</th><th>Category</th><th>Description</th></tr></thead><tbody>']
        for ev in events:
            step = ev.get('step', '')
            cat = ev.get('category', ev.get('type', ''))
            desc = ev.get('description', ev.get('message', str(ev)))
            parts.append(
                f'<tr><td>{_esc(str(step))}</td>'
                f'<td>{_esc(str(cat))}</td>'
                f'<td>{_esc(str(desc))}</td></tr>'
            )
        parts.append('</tbody></table></div>')
        return '\n'.join(parts)

    def _config_section(self, report: SimulationReport) -> str:
        cfg = report.config
        items = [
            ("Strategy", cfg.strategy),
            ("Max Depth", str(cfg.max_depth)),
            ("Max Replicas", str(cfg.max_replicas)),
            ("Tasks/Worker", str(cfg.tasks_per_worker)),
            ("Cooldown", f"{cfg.cooldown_seconds}s"),
            ("Repl. Probability", f"{cfg.replication_probability:.0%}"),
        ]
        parts = ['<h2>Configuration</h2>', '<div class="grid">']
        for label, value in items:
            parts.append(
                f'<div class="card">'
                f'<div class="card-label">{_esc(label)}</div>'
                f'<div class="card-value" style="font-size:1.1rem">{_esc(value)}</div>'
                f'</div>'
            )
        parts.append('</div>')
        return '\n'.join(parts)

    def _comparison_table(
        self, reports: List[SimulationReport], labels: List[str]
    ) -> str:
        metrics = [
            ("Strategy", lambda r: r.config.strategy),
            ("Workers", lambda r: str(len(r.workers))),
            ("Max Depth", lambda r: str(max((w.depth for w in r.workers.values()), default=0))),
            ("Tasks", lambda r: str(r.total_tasks)),
            ("Repl. Attempted", lambda r: str(r.total_replications_attempted)),
            ("Repl. Succeeded", lambda r: str(r.total_replications_succeeded)),
            ("Repl. Denied", lambda r: str(r.total_replications_denied)),
            ("Success Rate", lambda r: f"{r.total_replications_succeeded / max(1, r.total_replications_attempted) * 100:.1f}%"),
            ("Duration (ms)", lambda r: f"{r.duration_ms:.0f}"),
            ("Timeline Events", lambda r: str(len(r.timeline))),
        ]
        parts = ['<table><thead><tr><th>Metric</th>']
        for lbl in labels:
            parts.append(f'<th>{_esc(lbl)}</th>')
        parts.append('</tr></thead><tbody>')
        for name, fn in metrics:
            parts.append(f'<tr><td><strong>{_esc(name)}</strong></td>')
            for r in reports:
                parts.append(f'<td>{_esc(fn(r))}</td>')
            parts.append('</tr>')
        parts.append('</tbody></table>')
        return '\n'.join(parts)

    def _comparison_bars(
        self, reports: List[SimulationReport], labels: List[str]
    ) -> str:
        colors = ['#4f46e5', '#059669', '#d97706', '#dc2626', '#7c3aed', '#0891b2']
        # Show worker count bars
        max_workers = max(len(r.workers) for r in reports) or 1
        parts = ['<h2>Worker Count Comparison</h2>', '<div class="bar-chart">']
        for i, (r, lbl) in enumerate(zip(reports, labels)):
            count = len(r.workers)
            pct = count / max_workers * 100
            c = colors[i % len(colors)]
            parts.append(
                f'<div class="bar-row">'
                f'<span class="bar-label">{_esc(lbl)}</span>'
                f'<div class="bar-track"><div class="bar-fill" style="width:{pct:.0f}%;background:{c}">'
                f'{count}</div></div>'
                f'<span class="bar-val">{count}</span>'
                f'</div>'
            )
        parts.append('</div>')
        return '\n'.join(parts)

    def _tabbed_details(
        self, reports: List[SimulationReport], labels: List[str]
    ) -> str:
        group = "compare"
        parts = ['<div class="tab-bar">']
        for i, lbl in enumerate(labels):
            active = ' active' if i == 0 else ''
            parts.append(
                f'<button class="tab-btn{active}" data-group="{group}" '
                f'data-tab="tab-{group}-{i}">{_esc(lbl)}</button>'
            )
        parts.append('</div>')

        for i, (r, lbl) in enumerate(zip(reports, labels)):
            active = ' active' if i == 0 else ''
            parts.append(f'<div id="tab-{group}-{i}" class="tab-panel{active}" data-group="{group}">')
            parts.append(self._summary_cards(r))
            parts.append(self._worker_depth_chart(r))
            if self.config.show_tree:
                parts.append(f'<h3>Lineage Tree</h3><pre>{_esc(r.render_tree())}</pre>')
            parts.append('</div>')

        return '\n'.join(parts)

    # ── Public utilities ─────────────────────────────────────────

    def get_report_data(self, report: SimulationReport) -> Dict[str, Any]:
        """Extract structured data from a report for programmatic use."""
        workers = report.workers
        depth_counts: Dict[int, int] = {}
        for w in workers.values():
            depth_counts[w.depth] = depth_counts.get(w.depth, 0) + 1

        return {
            "strategy": report.config.strategy,
            "worker_count": len(workers),
            "max_depth": max((w.depth for w in workers.values()), default=0),
            "total_tasks": report.total_tasks,
            "replications_attempted": report.total_replications_attempted,
            "replications_succeeded": report.total_replications_succeeded,
            "replications_denied": report.total_replications_denied,
            "success_rate": (
                report.total_replications_succeeded / report.total_replications_attempted
                if report.total_replications_attempted > 0 else 0.0
            ),
            "duration_ms": report.duration_ms,
            "depth_distribution": depth_counts,
            "timeline_count": len(report.timeline),
            "audit_count": len(report.audit_events),
        }


# ── Helpers ──────────────────────────────────────────────────────

def _esc(text: str) -> str:
    return html_lib.escape(str(text))


def _depth_color(depth: int) -> str:
    colors = ['#4f46e5', '#2563eb', '#0891b2', '#059669', '#65a30d',
              '#d97706', '#ea580c', '#dc2626', '#9333ea', '#c026d3']
    return colors[depth % len(colors)]


def _event_badge(event_type: str) -> str:
    et = event_type.lower()
    if 'denied' in et or 'fail' in et or 'shutdown' in et:
        return 'badge-red'
    if 'success' in et or 'complet' in et or 'creat' in et:
        return 'badge-green'
    if 'warn' in et or 'limit' in et:
        return 'badge-yellow'
    return 'badge-blue'


# ── CLI ──────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate HTML dashboard for simulation runs"
    )
    parser.add_argument(
        "--strategy", default="greedy",
        help="Simulation strategy (default: greedy)"
    )
    parser.add_argument(
        "--compare", nargs="+", metavar="STRATEGY",
        help="Compare multiple strategies"
    )
    parser.add_argument("--depth", type=int, default=3, help="Max replication depth")
    parser.add_argument("--replicas", type=int, default=10, help="Max replicas per worker")
    parser.add_argument("--tasks", type=int, default=2, help="Tasks per worker")
    parser.add_argument("-o", "--output", default=None, help="Output file (default: stdout)")
    parser.add_argument("--dark", action="store_true", help="Dark theme")
    parser.add_argument("--no-tree", action="store_true", help="Omit lineage tree")
    parser.add_argument("--no-timeline", action="store_true", help="Omit timeline table")
    parser.add_argument("--json", action="store_true", help="Output JSON data instead of HTML")

    args = parser.parse_args(argv)

    config = DashboardConfig(
        theme="dark" if args.dark else "light",
        show_tree=not args.no_tree,
        show_timeline=not args.no_timeline,
    )
    gen = DashboardGenerator(config)

    strategies = args.compare if args.compare else [args.strategy]
    reports: List[SimulationReport] = []
    labels: List[str] = []

    for strategy in strategies:
        sim = Simulator(ScenarioConfig(
            strategy=strategy,
            max_depth=args.depth,
            max_replicas=args.replicas,
            tasks_per_worker=args.tasks,
        ))
        report = sim.run()
        reports.append(report)
        labels.append(strategy.capitalize())

    if args.json:
        data = [gen.get_report_data(r) for r in reports]
        output = json.dumps(data if len(data) > 1 else data[0], indent=2)
    elif len(reports) == 1:
        output = gen.single_report(reports[0], title=f"Simulation: {labels[0]}")
    else:
        output = gen.compare_reports(reports, labels)

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(output)
        print(f"Dashboard written to {args.output}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
