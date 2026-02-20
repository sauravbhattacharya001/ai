"""Interactive HTML report generator for simulation and threat analysis.

Produces self-contained HTML files with interactive Canvas charts,
collapsible sections, and dark/light theme toggle. Zero external
dependencies — everything is embedded.

Usage (CLI)::

    python -m replication.reporter --simulation             # simulation report
    python -m replication.reporter --threats                 # threat assessment
    python -m replication.reporter --comparison              # strategy comparison
    python -m replication.reporter --all                     # combined report
    python -m replication.reporter --all -o report.html      # save to file
    python -m replication.reporter --threats --open          # generate and open in browser

Programmatic::

    from replication.reporter import HTMLReporter
    from replication.simulator import Simulator
    reporter = HTMLReporter()

    # From a simulation
    sim = Simulator()
    report = sim.run()
    html = reporter.simulation_report(report)

    # From threat analysis
    from replication.threats import ThreatSimulator
    threats = ThreatSimulator()
    threat_report = threats.run_all()
    html = reporter.threat_report(threat_report)

    # Combined report
    html = reporter.combined_report(
        simulation=report,
        threat=threat_report,
    )

    # Save to file
    reporter.save(html, "report.html")
"""

from __future__ import annotations

import html as html_mod
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .simulator import SimulationReport, ScenarioConfig
from .comparator import ComparisonResult
from .threats import ThreatReport, MitigationStatus, ThreatSeverity


class HTMLReporter:
    """Generate self-contained interactive HTML reports."""

    def __init__(self) -> None:
        self._timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # ─── CSS ────────────────────────────────────────────────────────

    @staticmethod
    def _css() -> str:
        return """
        :root {
            --bg: #0d1117;
            --surface: #161b22;
            --surface2: #21262d;
            --border: #30363d;
            --text: #e6edf3;
            --text-muted: #8b949e;
            --accent: #58a6ff;
            --accent2: #3fb950;
            --danger: #f85149;
            --warning: #d29922;
            --info: #79c0ff;
            --radius: 8px;
            --shadow: 0 2px 8px rgba(0,0,0,0.3);
        }
        [data-theme="light"] {
            --bg: #f6f8fa;
            --surface: #ffffff;
            --surface2: #f0f3f6;
            --border: #d0d7de;
            --text: #1f2328;
            --text-muted: #656d76;
            --accent: #0969da;
            --accent2: #1a7f37;
            --danger: #cf222e;
            --warning: #9a6700;
            --info: #0550ae;
            --shadow: 0 2px 8px rgba(0,0,0,0.08);
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.6;
            padding: 24px;
            max-width: 1200px;
            margin: 0 auto;
            transition: background 0.3s, color 0.3s;
        }
        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 32px;
            padding-bottom: 16px;
            border-bottom: 1px solid var(--border);
        }
        header h1 {
            font-size: 1.8em;
            font-weight: 600;
        }
        header .meta { color: var(--text-muted); font-size: 0.85em; }
        .theme-toggle {
            background: var(--surface2);
            border: 1px solid var(--border);
            color: var(--text);
            padding: 6px 14px;
            border-radius: 20px;
            cursor: pointer;
            font-size: 0.85em;
            transition: all 0.2s;
        }
        .theme-toggle:hover { border-color: var(--accent); }
        .section {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            margin-bottom: 20px;
            box-shadow: var(--shadow);
            overflow: hidden;
        }
        .section-header {
            padding: 16px 20px;
            cursor: pointer;
            user-select: none;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-weight: 600;
            font-size: 1.1em;
            transition: background 0.2s;
        }
        .section-header:hover { background: var(--surface2); }
        .section-header .arrow {
            transition: transform 0.3s;
            font-size: 0.8em;
            color: var(--text-muted);
        }
        .section.collapsed .section-body { display: none; }
        .section.collapsed .arrow { transform: rotate(-90deg); }
        .section-body { padding: 0 20px 20px; }
        .cards {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 12px;
            margin-bottom: 20px;
        }
        .card {
            background: var(--surface2);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 16px;
            text-align: center;
        }
        .card .value {
            font-size: 2em;
            font-weight: 700;
            color: var(--accent);
        }
        .card .label {
            font-size: 0.8em;
            color: var(--text-muted);
            margin-top: 4px;
        }
        .chart-container {
            background: var(--surface2);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 16px;
            margin-bottom: 16px;
        }
        .chart-title {
            font-weight: 600;
            margin-bottom: 12px;
            font-size: 0.95em;
        }
        canvas {
            width: 100% !important;
            height: auto !important;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9em;
            margin-bottom: 16px;
        }
        th, td {
            padding: 10px 14px;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }
        th {
            background: var(--surface2);
            font-weight: 600;
            font-size: 0.85em;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        tr:hover { background: var(--surface2); }
        .badge {
            display: inline-block;
            padding: 2px 10px;
            border-radius: 12px;
            font-size: 0.8em;
            font-weight: 600;
        }
        .badge-green { background: #1a7f3722; color: var(--accent2); border: 1px solid #3fb95044; }
        .badge-red { background: #f8514922; color: var(--danger); border: 1px solid #f8514944; }
        .badge-yellow { background: #d2992222; color: var(--warning); border: 1px solid #d2992244; }
        .badge-blue { background: #58a6ff22; color: var(--accent); border: 1px solid #58a6ff44; }
        .progress-bar {
            width: 100%;
            height: 24px;
            background: var(--surface2);
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid var(--border);
            margin: 8px 0;
        }
        .progress-fill {
            height: 100%;
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.75em;
            font-weight: 700;
            color: white;
            transition: width 0.5s ease;
        }
        .tree-view {
            font-family: 'Cascadia Code', 'Fira Code', 'Consolas', monospace;
            font-size: 0.85em;
            background: var(--surface2);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 16px;
            overflow-x: auto;
            white-space: pre;
            line-height: 1.8;
        }
        .timeline-entry {
            display: flex;
            align-items: flex-start;
            gap: 12px;
            padding: 8px 0;
            border-bottom: 1px solid var(--border);
            font-size: 0.85em;
        }
        .timeline-entry:last-child { border-bottom: none; }
        .timeline-icon { font-size: 1.2em; min-width: 24px; text-align: center; }
        .timeline-time { color: var(--text-muted); min-width: 80px; font-family: monospace; }
        .timeline-detail { flex: 1; }
        .tooltip-container { position: relative; }
        .tab-bar {
            display: flex;
            gap: 4px;
            margin-bottom: 16px;
            border-bottom: 2px solid var(--border);
            padding-bottom: 0;
        }
        .tab-btn {
            padding: 8px 18px;
            border: none;
            background: none;
            color: var(--text-muted);
            cursor: pointer;
            font-size: 0.9em;
            border-bottom: 2px solid transparent;
            margin-bottom: -2px;
            transition: all 0.2s;
        }
        .tab-btn.active {
            color: var(--accent);
            border-bottom-color: var(--accent);
            font-weight: 600;
        }
        .tab-btn:hover { color: var(--text); }
        .tab-panel { display: none; }
        .tab-panel.active { display: block; }
        .grade-badge {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 48px;
            height: 48px;
            border-radius: 50%;
            font-size: 1.4em;
            font-weight: 800;
        }
        .grade-a { background: #1a7f3733; color: var(--accent2); border: 2px solid var(--accent2); }
        .grade-b { background: #58a6ff33; color: var(--accent); border: 2px solid var(--accent); }
        .grade-c { background: #d2992233; color: var(--warning); border: 2px solid var(--warning); }
        .grade-d, .grade-f { background: #f8514933; color: var(--danger); border: 2px solid var(--danger); }
        @media (max-width: 768px) {
            body { padding: 12px; }
            .cards { grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); }
            header h1 { font-size: 1.3em; }
        }
        """

    # ─── JavaScript ─────────────────────────────────────────────────

    @staticmethod
    def _js() -> str:
        return """
        // Theme toggle
        function toggleTheme() {
            const body = document.documentElement;
            const current = body.getAttribute('data-theme') || 'dark';
            const next = current === 'dark' ? 'light' : 'dark';
            body.setAttribute('data-theme', next);
            document.querySelector('.theme-toggle').textContent = next === 'dark' ? '☀️ Light' : '🌙 Dark';
            // Redraw all charts
            Object.keys(window._charts || {}).forEach(k => window._charts[k]());
        }

        // Collapsible sections
        document.addEventListener('click', e => {
            const hdr = e.target.closest('.section-header');
            if (hdr) hdr.parentElement.classList.toggle('collapsed');
        });

        // Tabs
        function switchTab(group, tabId) {
            document.querySelectorAll(`[data-tab-group="${group}"] .tab-btn`).forEach(b => {
                b.classList.toggle('active', b.dataset.tab === tabId);
            });
            document.querySelectorAll(`[data-tab-group="${group}"] .tab-panel`).forEach(p => {
                p.classList.toggle('active', p.id === tabId);
            });
        }

        // Chart helpers
        window._charts = {};

        function getThemeColors() {
            const style = getComputedStyle(document.documentElement);
            return {
                text: style.getPropertyValue('--text').trim(),
                muted: style.getPropertyValue('--text-muted').trim(),
                accent: style.getPropertyValue('--accent').trim(),
                accent2: style.getPropertyValue('--accent2').trim(),
                danger: style.getPropertyValue('--danger').trim(),
                warning: style.getPropertyValue('--warning').trim(),
                info: style.getPropertyValue('--info').trim(),
                border: style.getPropertyValue('--border').trim(),
                surface: style.getPropertyValue('--surface').trim(),
                surface2: style.getPropertyValue('--surface2').trim(),
            };
        }

        const PALETTE = [
            '#58a6ff', '#3fb950', '#f85149', '#d29922',
            '#bc8cff', '#79c0ff', '#f778ba', '#56d4dd',
            '#e3b341', '#7ee787',
        ];

        function drawBarChart(canvasId, labels, datasets, options = {}) {
            const draw = () => {
                const canvas = document.getElementById(canvasId);
                if (!canvas) return;
                const ctx = canvas.getContext('2d');
                const dpr = window.devicePixelRatio || 1;
                const rect = canvas.parentElement.getBoundingClientRect();
                const W = rect.width - 32;
                const H = options.height || 280;
                canvas.width = W * dpr;
                canvas.height = H * dpr;
                canvas.style.width = W + 'px';
                canvas.style.height = H + 'px';
                ctx.scale(dpr, dpr);

                const c = getThemeColors();
                const pad = { top: 20, right: 20, bottom: 50, left: 60 };
                const chartW = W - pad.left - pad.right;
                const chartH = H - pad.top - pad.bottom;

                // Find max value
                let maxVal = 0;
                datasets.forEach(ds => ds.data.forEach(v => { if (v > maxVal) maxVal = v; }));
                if (maxVal === 0) maxVal = 1;
                maxVal *= 1.1;

                // Y axis
                ctx.strokeStyle = c.border;
                ctx.lineWidth = 1;
                ctx.beginPath();
                ctx.moveTo(pad.left, pad.top);
                ctx.lineTo(pad.left, pad.top + chartH);
                ctx.lineTo(pad.left + chartW, pad.top + chartH);
                ctx.stroke();

                // Y grid + labels
                const ySteps = 5;
                ctx.fillStyle = c.muted;
                ctx.font = '11px -apple-system, sans-serif';
                ctx.textAlign = 'right';
                for (let i = 0; i <= ySteps; i++) {
                    const y = pad.top + chartH - (i / ySteps) * chartH;
                    const val = (maxVal * i / ySteps);
                    ctx.fillText(val % 1 === 0 ? val.toString() : val.toFixed(1), pad.left - 8, y + 4);
                    if (i > 0) {
                        ctx.strokeStyle = c.border;
                        ctx.setLineDash([3, 3]);
                        ctx.beginPath();
                        ctx.moveTo(pad.left, y);
                        ctx.lineTo(pad.left + chartW, y);
                        ctx.stroke();
                        ctx.setLineDash([]);
                    }
                }

                // Bars
                const groupCount = labels.length;
                const barGroupW = chartW / groupCount;
                const barW = Math.min(barGroupW * 0.6 / datasets.length, 40);
                const groupOffset = (barGroupW - barW * datasets.length) / 2;

                datasets.forEach((ds, di) => {
                    const color = ds.color || PALETTE[di % PALETTE.length];
                    ctx.fillStyle = color;
                    ds.data.forEach((val, i) => {
                        const x = pad.left + i * barGroupW + groupOffset + di * barW;
                        const barH = (val / maxVal) * chartH;
                        const y = pad.top + chartH - barH;
                        ctx.beginPath();
                        ctx.roundRect(x, y, barW - 2, barH, [4, 4, 0, 0]);
                        ctx.fill();

                        // Value on top
                        if (barH > 20) {
                            ctx.fillStyle = c.text;
                            ctx.font = '10px -apple-system, sans-serif';
                            ctx.textAlign = 'center';
                            ctx.fillText(val % 1 === 0 ? val.toString() : val.toFixed(1),
                                         x + (barW - 2) / 2, y - 4);
                            ctx.fillStyle = color;
                        }
                    });
                });

                // X labels
                ctx.fillStyle = c.muted;
                ctx.font = '11px -apple-system, sans-serif';
                ctx.textAlign = 'center';
                labels.forEach((label, i) => {
                    const x = pad.left + i * barGroupW + barGroupW / 2;
                    ctx.fillText(label.length > 12 ? label.slice(0, 11) + '…' : label,
                                 x, pad.top + chartH + 20);
                });

                // Legend
                if (datasets.length > 1) {
                    let lx = pad.left;
                    const ly = pad.top + chartH + 36;
                    ctx.font = '11px -apple-system, sans-serif';
                    datasets.forEach((ds, di) => {
                        const color = ds.color || PALETTE[di % PALETTE.length];
                        ctx.fillStyle = color;
                        ctx.fillRect(lx, ly, 12, 12);
                        ctx.fillStyle = c.text;
                        ctx.textAlign = 'left';
                        ctx.fillText(ds.label, lx + 16, ly + 10);
                        lx += ctx.measureText(ds.label).width + 32;
                    });
                }
            };
            window._charts[canvasId] = draw;
            draw();
        }

        function drawDonutChart(canvasId, labels, values, colors) {
            const draw = () => {
                const canvas = document.getElementById(canvasId);
                if (!canvas) return;
                const ctx = canvas.getContext('2d');
                const dpr = window.devicePixelRatio || 1;
                const size = Math.min(canvas.parentElement.getBoundingClientRect().width - 32, 300);
                canvas.width = size * dpr;
                canvas.height = size * dpr;
                canvas.style.width = size + 'px';
                canvas.style.height = size + 'px';
                ctx.scale(dpr, dpr);

                const c = getThemeColors();
                const cx = size / 2;
                const cy = size / 2;
                const outerR = size / 2 - 20;
                const innerR = outerR * 0.55;
                const total = values.reduce((a, b) => a + b, 0);
                if (total === 0) return;

                let startAngle = -Math.PI / 2;
                values.forEach((val, i) => {
                    const sliceAngle = (val / total) * Math.PI * 2;
                    ctx.beginPath();
                    ctx.arc(cx, cy, outerR, startAngle, startAngle + sliceAngle);
                    ctx.arc(cx, cy, innerR, startAngle + sliceAngle, startAngle, true);
                    ctx.closePath();
                    ctx.fillStyle = colors[i] || PALETTE[i];
                    ctx.fill();

                    // Label
                    if (val > 0) {
                        const midAngle = startAngle + sliceAngle / 2;
                        const labelR = (outerR + innerR) / 2;
                        const lx = cx + Math.cos(midAngle) * labelR;
                        const ly = cy + Math.sin(midAngle) * labelR;
                        ctx.fillStyle = '#fff';
                        ctx.font = 'bold 12px -apple-system, sans-serif';
                        ctx.textAlign = 'center';
                        ctx.textBaseline = 'middle';
                        if (sliceAngle > 0.3) ctx.fillText(val.toString(), lx, ly);
                    }
                    startAngle += sliceAngle;
                });

                // Center text
                ctx.fillStyle = c.text;
                ctx.font = 'bold 24px -apple-system, sans-serif';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText(total.toString(), cx, cy - 6);
                ctx.font = '11px -apple-system, sans-serif';
                ctx.fillStyle = c.muted;
                ctx.fillText('total', cx, cy + 14);

                // Legend below
                let ly = size - 4;
                ctx.font = '11px -apple-system, sans-serif';
                ctx.textAlign = 'left';
                let lx = 10;
                labels.forEach((label, i) => {
                    ctx.fillStyle = colors[i] || PALETTE[i];
                    ctx.fillRect(lx, ly - 8, 10, 10);
                    ctx.fillStyle = c.text;
                    ctx.fillText(`${label} (${values[i]})`, lx + 14, ly);
                    lx += ctx.measureText(`${label} (${values[i]})`).width + 28;
                });
            };
            window._charts[canvasId] = draw;
            draw();
        }

        // Resize handler
        let resizeTimer;
        window.addEventListener('resize', () => {
            clearTimeout(resizeTimer);
            resizeTimer = setTimeout(() => {
                Object.values(window._charts).forEach(fn => fn());
            }, 200);
        });
        """

    # ─── HTML helpers ───────────────────────────────────────────────

    def _esc(self, text: str) -> str:
        return html_mod.escape(str(text))

    def _wrap_page(self, title: str, body: str) -> str:
        return f"""<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{self._esc(title)}</title>
<style>{self._css()}</style>
</head>
<body>
<header>
  <div>
    <h1>🧬 {self._esc(title)}</h1>
    <div class="meta">Generated {self._timestamp} · AI Replication Sandbox</div>
  </div>
  <button class="theme-toggle" onclick="toggleTheme()">☀️ Light</button>
</header>
{body}
<script>{self._js()}</script>
</body>
</html>"""

    def _section(self, title: str, body: str, collapsed: bool = False,
                 icon: str = "") -> str:
        cls = " collapsed" if collapsed else ""
        prefix = f"{icon} " if icon else ""
        return f"""<div class="section{cls}">
  <div class="section-header">
    <span>{prefix}{self._esc(title)}</span>
    <span class="arrow">▼</span>
  </div>
  <div class="section-body">{body}</div>
</div>"""

    def _card(self, value: str, label: str, color: str = "") -> str:
        style = f' style="color:{color}"' if color else ''
        return f"""<div class="card">
  <div class="value"{style}>{self._esc(value)}</div>
  <div class="label">{self._esc(label)}</div>
</div>"""

    def _badge(self, text: str, kind: str = "blue") -> str:
        return f'<span class="badge badge-{kind}">{self._esc(text)}</span>'

    def _progress(self, pct: float, color: str = "#58a6ff", label: str = "") -> str:
        pct = max(0, min(100, pct))
        display = label or f"{pct:.0f}%"
        return f"""<div class="progress-bar">
  <div class="progress-fill" style="width:{pct:.1f}%;background:{color}">{self._esc(display)}</div>
</div>"""

    # ─── Simulation report ──────────────────────────────────────────

    def simulation_report(self, report: SimulationReport) -> str:
        """Generate an interactive HTML report from a SimulationReport."""
        body_parts: List[str] = []

        # Overview cards
        max_depth_reached = max(
            (w.depth for w in report.workers.values()), default=0
        )
        cards = '<div class="cards">'
        cards += self._card(str(len(report.workers)), "Workers Spawned")
        cards += self._card(str(report.total_tasks), "Tasks Completed")
        cards += self._card(str(report.total_replications_succeeded), "Replications OK", "#3fb950")
        cards += self._card(str(report.total_replications_denied), "Denied", "#f85149")
        cards += self._card(str(max_depth_reached), "Max Depth")
        cards += self._card(f"{report.duration_ms:.1f}ms", "Duration")
        cards += '</div>'

        config_table = self._config_table(report.config)
        body_parts.append(self._section("Overview", cards + config_table, icon="📊"))

        # Depth distribution chart
        depth_counts: Dict[int, int] = {}
        for w in report.workers.values():
            depth_counts[w.depth] = depth_counts.get(w.depth, 0) + 1
        depths = sorted(depth_counts.keys())
        depth_labels = [f"Depth {d}" for d in depths]
        depth_values = [depth_counts[d] for d in depths]

        chart_html = '<div class="chart-container">'
        chart_html += '<div class="chart-title">Worker Depth Distribution</div>'
        chart_html += '<canvas id="depthChart"></canvas>'
        chart_html += '</div>'
        chart_html += f"""<script>
        drawBarChart('depthChart',
            {json.dumps(depth_labels)},
            [{{label: 'Workers', data: {json.dumps(depth_values)}, color: '#58a6ff'}}]
        );
        </script>"""

        # Replication outcome chart
        chart_html += '<div class="chart-container">'
        chart_html += '<div class="chart-title">Replication Outcomes</div>'
        chart_html += '<canvas id="replChart"></canvas>'
        chart_html += '</div>'
        chart_html += f"""<script>
        drawDonutChart('replChart',
            ['Succeeded', 'Denied'],
            [{report.total_replications_succeeded}, {report.total_replications_denied}],
            ['#3fb950', '#f85149']
        );
        </script>"""

        body_parts.append(self._section("Charts", chart_html, icon="📈"))

        # Worker tree
        tree_text = report.render_tree()
        tree_html = f'<div class="tree-view">{self._esc(tree_text)}</div>'
        body_parts.append(self._section("Worker Tree", tree_html, icon="🌳"))

        # Timeline
        timeline_html = self._render_timeline_html(report.timeline)
        body_parts.append(self._section("Timeline", timeline_html, collapsed=True, icon="⏱️"))

        # Worker table
        worker_table = self._worker_table(report)
        body_parts.append(self._section("Worker Details", worker_table, collapsed=True, icon="📋"))

        return self._wrap_page("Simulation Report", "\n".join(body_parts))

    def _config_table(self, config: ScenarioConfig) -> str:
        rows = [
            ("Strategy", config.strategy),
            ("Max Depth", config.max_depth),
            ("Max Replicas", config.max_replicas),
            ("Cooldown", f"{config.cooldown_seconds}s"),
            ("Tasks/Worker", config.tasks_per_worker),
            ("CPU Limit", f"{config.cpu_limit} cores"),
            ("Memory Limit", f"{config.memory_limit_mb} MB"),
        ]
        html = '<table><thead><tr><th>Parameter</th><th>Value</th></tr></thead><tbody>'
        for label, val in rows:
            html += f'<tr><td>{self._esc(label)}</td><td>{self._esc(str(val))}</td></tr>'
        html += '</tbody></table>'
        return html

    def _worker_table(self, report: SimulationReport) -> str:
        html = '<table><thead><tr>'
        html += '<th>Worker ID</th><th>Depth</th><th>Parent</th>'
        html += '<th>Tasks</th><th>Repl OK</th><th>Denied</th><th>Children</th>'
        html += '</tr></thead><tbody>'
        for wid, rec in sorted(report.workers.items(), key=lambda x: x[1].depth):
            parent = rec.parent_id or '—'
            html += f'<tr>'
            html += f'<td><code>{self._esc(wid)}</code></td>'
            html += f'<td>{rec.depth}</td>'
            html += f'<td><code>{self._esc(parent)}</code></td>'
            html += f'<td>{rec.tasks_completed}</td>'
            html += f'<td>{rec.replications_succeeded}</td>'
            html += f'<td>{rec.replications_denied}</td>'
            html += f'<td>{len(rec.children)}</td>'
            html += f'</tr>'
        html += '</tbody></table>'
        return html

    def _render_timeline_html(self, timeline: List[Dict[str, Any]]) -> str:
        icons = {
            "spawn": "🟢",
            "task": "⚡",
            "replicate_ok": "🔀",
            "replicate_denied": "🚫",
            "shutdown": "🔴",
        }
        html = '<div class="timeline">'
        # Show at most 100 entries, with a note if truncated
        display_items = timeline[:100]
        for entry in display_items:
            icon = icons.get(entry.get("type", ""), "•")
            time_ms = entry.get("time_ms", 0)
            wid = entry.get("worker_id", "?")[:8]
            detail = entry.get("detail", "")
            html += f"""<div class="timeline-entry">
  <span class="timeline-icon">{icon}</span>
  <span class="timeline-time">{time_ms:.1f}ms</span>
  <span class="timeline-detail"><code>[{self._esc(wid)}]</code> {self._esc(detail)}</span>
</div>"""
        if len(timeline) > 100:
            html += f'<div class="timeline-entry"><span class="timeline-icon">…</span>'
            html += f'<span class="timeline-detail">{len(timeline) - 100} more events omitted</span></div>'
        html += '</div>'
        return html

    # ─── Threat report ──────────────────────────────────────────────

    def threat_report(self, report: ThreatReport) -> str:
        """Generate an interactive HTML report from a ThreatReport."""
        body_parts: List[str] = []

        # Score overview
        score = report.security_score
        grade = report.grade
        grade_class = "grade-a" if grade.startswith("A") else \
                      "grade-b" if grade.startswith("B") else \
                      "grade-c" if grade.startswith("C") else "grade-d"
        score_color = "#3fb950" if score >= 80 else "#d29922" if score >= 60 else "#f85149"

        overview = f'<div style="display:flex;align-items:center;gap:20px;margin-bottom:20px">'
        overview += f'<span class="grade-badge {grade_class}">{self._esc(grade)}</span>'
        overview += f'<div style="flex:1">'
        overview += f'<div style="font-weight:600;margin-bottom:4px">Security Score: {score:.0f}/100</div>'
        overview += self._progress(score, score_color)
        overview += f'</div></div>'

        cards = '<div class="cards">'
        cards += self._card(str(len(report.results)), "Scenarios Tested")
        cards += self._card(str(report.total_mitigated), "Mitigated", "#3fb950")
        cards += self._card(str(report.total_partial), "Partial", "#d29922")
        cards += self._card(str(report.total_failed), "Failed", "#f85149")
        cards += self._card(f"{report.duration_ms:.1f}ms", "Duration")
        cards += '</div>'

        body_parts.append(self._section("Security Overview", overview + cards, icon="🛡️"))

        # Threat matrix chart
        threat_labels = [r.name[:20] for r in report.results]
        block_rates = [r.block_rate for r in report.results]

        chart_html = '<div class="chart-container">'
        chart_html += '<div class="chart-title">Block Rate by Threat Scenario</div>'
        chart_html += '<canvas id="threatChart"></canvas>'
        chart_html += '</div>'

        colors_js = []
        for r in report.results:
            if r.status == MitigationStatus.MITIGATED:
                colors_js.append("#3fb950")
            elif r.status == MitigationStatus.PARTIAL:
                colors_js.append("#d29922")
            else:
                colors_js.append("#f85149")

        chart_html += f"""<script>
        (function() {{
            const canvas = document.getElementById('threatChart');
            const draw = () => {{
                if (!canvas) return;
                const ctx = canvas.getContext('2d');
                const dpr = window.devicePixelRatio || 1;
                const W = canvas.parentElement.getBoundingClientRect().width - 32;
                const H = 280;
                canvas.width = W * dpr;
                canvas.height = H * dpr;
                canvas.style.width = W + 'px';
                canvas.style.height = H + 'px';
                ctx.scale(dpr, dpr);
                const c = getThemeColors();
                const labels = {json.dumps(threat_labels)};
                const values = {json.dumps(block_rates)};
                const colors = {json.dumps(colors_js)};
                const pad = {{top: 20, right: 20, bottom: 60, left: 60}};
                const chartW = W - pad.left - pad.right;
                const chartH = H - pad.top - pad.bottom;
                // Y axis
                ctx.strokeStyle = c.border;
                ctx.beginPath();
                ctx.moveTo(pad.left, pad.top);
                ctx.lineTo(pad.left, pad.top + chartH);
                ctx.lineTo(pad.left + chartW, pad.top + chartH);
                ctx.stroke();
                // Y grid
                for (let i = 0; i <= 4; i++) {{
                    const y = pad.top + chartH - (i / 4) * chartH;
                    ctx.fillStyle = c.muted;
                    ctx.font = '11px sans-serif';
                    ctx.textAlign = 'right';
                    ctx.fillText((i * 25) + '%', pad.left - 8, y + 4);
                    ctx.strokeStyle = c.border;
                    ctx.setLineDash([3,3]);
                    ctx.beginPath();
                    ctx.moveTo(pad.left, y);
                    ctx.lineTo(pad.left + chartW, y);
                    ctx.stroke();
                    ctx.setLineDash([]);
                }}
                // Bars
                const barW = Math.min(chartW / labels.length * 0.7, 50);
                const gap = chartW / labels.length;
                values.forEach((val, i) => {{
                    const x = pad.left + i * gap + (gap - barW) / 2;
                    const barH = (val / 100) * chartH;
                    const y = pad.top + chartH - barH;
                    ctx.fillStyle = colors[i];
                    ctx.beginPath();
                    ctx.roundRect(x, y, barW, barH, [4,4,0,0]);
                    ctx.fill();
                    ctx.fillStyle = c.text;
                    ctx.font = 'bold 11px sans-serif';
                    ctx.textAlign = 'center';
                    ctx.fillText(val.toFixed(0) + '%', x + barW/2, y - 5);
                }});
                // X labels
                ctx.fillStyle = c.muted;
                ctx.font = '10px sans-serif';
                labels.forEach((l, i) => {{
                    const x = pad.left + i * gap + gap / 2;
                    ctx.save();
                    ctx.translate(x, pad.top + chartH + 10);
                    ctx.rotate(Math.PI / 6);
                    ctx.textAlign = 'left';
                    ctx.fillText(l, 0, 0);
                    ctx.restore();
                }});
            }};
            window._charts['threatChart'] = draw;
            draw();
        }})();
        </script>"""

        # Mitigation status donut
        chart_html += '<div class="chart-container">'
        chart_html += '<div class="chart-title">Mitigation Status</div>'
        chart_html += '<canvas id="mitigationDonut"></canvas>'
        chart_html += '</div>'
        chart_html += f"""<script>
        drawDonutChart('mitigationDonut',
            ['Mitigated', 'Partial', 'Failed'],
            [{report.total_mitigated}, {report.total_partial}, {report.total_failed}],
            ['#3fb950', '#d29922', '#f85149']
        );
        </script>"""

        body_parts.append(self._section("Analysis Charts", chart_html, icon="📊"))

        # Threat details table
        threat_table = '<table><thead><tr>'
        threat_table += '<th>Threat</th><th>Severity</th><th>Status</th>'
        threat_table += '<th>Block Rate</th><th>Attacks</th><th>Details</th>'
        threat_table += '</tr></thead><tbody>'

        severity_badges = {
            ThreatSeverity.CRITICAL: ("CRITICAL", "red"),
            ThreatSeverity.HIGH: ("HIGH", "yellow"),
            ThreatSeverity.MEDIUM: ("MEDIUM", "blue"),
            ThreatSeverity.LOW: ("LOW", "green"),
        }
        status_badges = {
            MitigationStatus.MITIGATED: ("✅ PASS", "green"),
            MitigationStatus.PARTIAL: ("⚠️ WARN", "yellow"),
            MitigationStatus.FAILED: ("❌ FAIL", "red"),
        }

        for r in report.results:
            sev_text, sev_kind = severity_badges[r.severity]
            stat_text, stat_kind = status_badges[r.status]
            details_html = '<br>'.join(self._esc(d) for d in r.details[:3])
            if len(r.details) > 3:
                details_html += f'<br><em>+{len(r.details) - 3} more</em>'

            threat_table += f'<tr>'
            threat_table += f'<td><strong>{self._esc(r.name)}</strong><br>'
            threat_table += f'<span style="color:var(--text-muted);font-size:0.8em">{self._esc(r.description)}</span></td>'
            threat_table += f'<td>{self._badge(sev_text, sev_kind)}</td>'
            threat_table += f'<td>{self._badge(stat_text, stat_kind)}</td>'
            threat_table += f'<td>{self._progress(r.block_rate, "#3fb950" if r.block_rate >= 80 else "#d29922" if r.block_rate >= 50 else "#f85149")}</td>'
            threat_table += f'<td>{r.attacks_blocked}/{r.attacks_attempted}</td>'
            threat_table += f'<td style="font-size:0.8em">{details_html}</td>'
            threat_table += f'</tr>'

        threat_table += '</tbody></table>'
        body_parts.append(self._section("Threat Details", threat_table, icon="⚔️"))

        # Recommendations
        recs_html = report.render_recommendations()
        if recs_html.strip():
            recs_pre = f'<div class="tree-view">{self._esc(recs_html)}</div>'
            body_parts.append(self._section("Recommendations", recs_pre, icon="💡"))

        return self._wrap_page("Threat Assessment Report", "\n".join(body_parts))

    # ─── Comparison report ──────────────────────────────────────────

    def comparison_report(self, result: ComparisonResult) -> str:
        """Generate an interactive HTML report from a ComparisonResult."""
        body_parts: List[str] = []

        # Overview
        overview = f'<p style="color:var(--text-muted);margin-bottom:16px">'
        overview += f'Comparing {len(result.runs)} scenarios'
        if result.swept_param:
            overview += f' (sweep: {self._esc(result.swept_param)})'
        overview += '</p>'

        metrics = result._metric_table()

        # Key metric cards from best performer
        if metrics:
            best_workers = max(metrics, key=lambda r: r["workers"])
            best_eff = max(metrics, key=lambda r: r["efficiency"])
            cards = '<div class="cards">'
            cards += self._card(str(len(result.runs)), "Scenarios")
            cards += self._card(best_workers["label"], "Most Workers", "#58a6ff")
            cards += self._card(best_eff["label"], "Most Efficient", "#3fb950")
            cards += self._card(
                f"{max(m['success_rate'] for m in metrics):.0f}%",
                "Best Replication Rate"
            )
            cards += '</div>'
            overview += cards

        body_parts.append(self._section("Comparison Overview", overview, icon="⚖️"))

        # Multi-metric bar chart
        if metrics:
            labels_js = json.dumps([m["label"] for m in metrics])
            workers_js = json.dumps([m["workers"] for m in metrics])
            tasks_js = json.dumps([m["tasks"] for m in metrics])
            repl_ok_js = json.dumps([m["repl_ok"] for m in metrics])
            denied_js = json.dumps([m["repl_denied"] for m in metrics])

            chart_html = '<div class="chart-container">'
            chart_html += '<div class="chart-title">Workers &amp; Tasks by Scenario</div>'
            chart_html += '<canvas id="compChart1"></canvas>'
            chart_html += '</div>'
            chart_html += f"""<script>
            drawBarChart('compChart1', {labels_js}, [
                {{label: 'Workers', data: {workers_js}, color: '#58a6ff'}},
                {{label: 'Tasks', data: {tasks_js}, color: '#3fb950'}}
            ]);
            </script>"""

            chart_html += '<div class="chart-container">'
            chart_html += '<div class="chart-title">Replication Outcomes</div>'
            chart_html += '<canvas id="compChart2"></canvas>'
            chart_html += '</div>'
            chart_html += f"""<script>
            drawBarChart('compChart2', {labels_js}, [
                {{label: 'Succeeded', data: {repl_ok_js}, color: '#3fb950'}},
                {{label: 'Denied', data: {denied_js}, color: '#f85149'}}
            ]);
            </script>"""

            # Efficiency chart
            eff_js = json.dumps([round(m["efficiency"], 2) for m in metrics])
            chart_html += '<div class="chart-container">'
            chart_html += '<div class="chart-title">Task Efficiency (tasks/worker)</div>'
            chart_html += '<canvas id="compChart3"></canvas>'
            chart_html += '</div>'
            chart_html += f"""<script>
            drawBarChart('compChart3', {labels_js}, [
                {{label: 'Efficiency', data: {eff_js}, color: '#bc8cff'}}
            ]);
            </script>"""

            body_parts.append(self._section("Comparison Charts", chart_html, icon="📈"))

        # Metrics table
        if metrics:
            table = '<table><thead><tr>'
            table += '<th>Scenario</th><th>Workers</th><th>Tasks</th>'
            table += '<th>Repl OK</th><th>Denied</th><th>Success %</th>'
            table += '<th>Max Depth</th><th>Efficiency</th><th>Duration</th>'
            table += '</tr></thead><tbody>'
            for m in metrics:
                table += '<tr>'
                table += f'<td><strong>{self._esc(m["label"])}</strong></td>'
                table += f'<td>{m["workers"]}</td>'
                table += f'<td>{m["tasks"]}</td>'
                table += f'<td>{m["repl_ok"]}</td>'
                table += f'<td>{m["repl_denied"]}</td>'
                table += f'<td>{m["success_rate"]:.1f}%</td>'
                table += f'<td>{m["max_depth"]}</td>'
                table += f'<td>{m["efficiency"]:.2f}</td>'
                table += f'<td>{m["duration_ms"]:.1f}ms</td>'
                table += '</tr>'
            table += '</tbody></table>'
            body_parts.append(self._section("Metrics Table", table, icon="📋"))

        # Insights (text)
        insights = result.render_insights()
        if insights:
            body_parts.append(self._section(
                "Insights",
                f'<div class="tree-view">{self._esc(insights)}</div>',
                icon="💡"
            ))

        return self._wrap_page(f"Comparison: {result.title}", "\n".join(body_parts))

    # ─── Combined report ────────────────────────────────────────────

    def combined_report(
        self,
        simulation: Optional[SimulationReport] = None,
        threat: Optional[ThreatReport] = None,
        comparison: Optional[ComparisonResult] = None,
    ) -> str:
        """Generate a combined HTML report with tabs for each section."""
        tabs: List[tuple] = []  # (id, label, content)

        if simulation:
            # Re-use internal rendering but extract just the body
            sim_body = self._sim_body(simulation)
            tabs.append(("sim", "🧬 Simulation", sim_body))

        if threat:
            threat_body = self._threat_body(threat)
            tabs.append(("threat", "🛡️ Threats", threat_body))

        if comparison:
            comp_body = self._comp_body(comparison)
            tabs.append(("comp", "⚖️ Comparison", comp_body))

        if not tabs:
            return self._wrap_page("Empty Report", "<p>No data provided.</p>")

        # Tab bar
        tab_bar = '<div data-tab-group="main"><div class="tab-bar">'
        for i, (tid, label, _) in enumerate(tabs):
            active = " active" if i == 0 else ""
            tab_bar += f'<button class="tab-btn{active}" data-tab="{tid}" '
            tab_bar += f'onclick="switchTab(\'main\',\'{tid}\')">{label}</button>'
        tab_bar += '</div>'

        # Tab panels
        for i, (tid, _, content) in enumerate(tabs):
            active = " active" if i == 0 else ""
            tab_bar += f'<div class="tab-panel{active}" id="{tid}">{content}</div>'
        tab_bar += '</div>'

        return self._wrap_page("AI Replication Safety Report", tab_bar)

    def _sim_body(self, report: SimulationReport) -> str:
        """Build simulation section body (reusable for combined report)."""
        # This duplicates some logic from simulation_report but returns
        # just the inner HTML without the page wrapper.
        parts: List[str] = []

        max_depth_reached = max(
            (w.depth for w in report.workers.values()), default=0
        )
        cards = '<div class="cards">'
        cards += self._card(str(len(report.workers)), "Workers")
        cards += self._card(str(report.total_tasks), "Tasks")
        cards += self._card(str(report.total_replications_succeeded), "Repl OK", "#3fb950")
        cards += self._card(str(report.total_replications_denied), "Denied", "#f85149")
        cards += self._card(str(max_depth_reached), "Max Depth")
        cards += self._card(f"{report.duration_ms:.1f}ms", "Duration")
        cards += '</div>'
        parts.append(cards)
        parts.append(self._config_table(report.config))

        # Depth chart with unique ID
        depth_counts: Dict[int, int] = {}
        for w in report.workers.values():
            depth_counts[w.depth] = depth_counts.get(w.depth, 0) + 1
        depths = sorted(depth_counts.keys())
        parts.append('<div class="chart-container">')
        parts.append('<div class="chart-title">Worker Depth Distribution</div>')
        parts.append('<canvas id="simDepthChart"></canvas></div>')
        parts.append(f"""<script>
        drawBarChart('simDepthChart',
            {json.dumps([f"Depth {d}" for d in depths])},
            [{{label: 'Workers', data: {json.dumps([depth_counts[d] for d in depths])}, color: '#58a6ff'}}]
        );
        </script>""")

        # Tree
        tree_text = report.render_tree()
        parts.append(f'<div class="tree-view">{self._esc(tree_text)}</div>')

        return "\n".join(parts)

    def _threat_body(self, report: ThreatReport) -> str:
        """Build threat section body (reusable for combined report)."""
        parts: List[str] = []

        score = report.security_score
        grade = report.grade
        grade_class = "grade-a" if grade.startswith("A") else \
                      "grade-b" if grade.startswith("B") else \
                      "grade-c" if grade.startswith("C") else "grade-d"
        score_color = "#3fb950" if score >= 80 else "#d29922" if score >= 60 else "#f85149"

        parts.append(f'<div style="display:flex;align-items:center;gap:20px;margin-bottom:20px">')
        parts.append(f'<span class="grade-badge {grade_class}">{self._esc(grade)}</span>')
        parts.append(f'<div style="flex:1">')
        parts.append(f'<div style="font-weight:600">Security Score: {score:.0f}/100</div>')
        parts.append(self._progress(score, score_color))
        parts.append(f'</div></div>')

        cards = '<div class="cards">'
        cards += self._card(str(report.total_mitigated), "Mitigated", "#3fb950")
        cards += self._card(str(report.total_partial), "Partial", "#d29922")
        cards += self._card(str(report.total_failed), "Failed", "#f85149")
        cards += '</div>'
        parts.append(cards)

        # Threat donut
        parts.append('<div class="chart-container">')
        parts.append('<div class="chart-title">Mitigation Status</div>')
        parts.append('<canvas id="combThreatDonut"></canvas></div>')
        parts.append(f"""<script>
        drawDonutChart('combThreatDonut',
            ['Mitigated', 'Partial', 'Failed'],
            [{report.total_mitigated}, {report.total_partial}, {report.total_failed}],
            ['#3fb950', '#d29922', '#f85149']
        );
        </script>""")

        # Results table
        severity_badges = {
            ThreatSeverity.CRITICAL: ("CRITICAL", "red"),
            ThreatSeverity.HIGH: ("HIGH", "yellow"),
            ThreatSeverity.MEDIUM: ("MEDIUM", "blue"),
            ThreatSeverity.LOW: ("LOW", "green"),
        }
        status_badges = {
            MitigationStatus.MITIGATED: ("✅ PASS", "green"),
            MitigationStatus.PARTIAL: ("⚠️ WARN", "yellow"),
            MitigationStatus.FAILED: ("❌ FAIL", "red"),
        }

        table = '<table><thead><tr><th>Threat</th><th>Severity</th><th>Status</th><th>Block Rate</th></tr></thead><tbody>'
        for r in report.results:
            sev_text, sev_kind = severity_badges[r.severity]
            stat_text, stat_kind = status_badges[r.status]
            table += f'<tr><td>{self._esc(r.name)}</td>'
            table += f'<td>{self._badge(sev_text, sev_kind)}</td>'
            table += f'<td>{self._badge(stat_text, stat_kind)}</td>'
            table += f'<td>{r.block_rate:.0f}%</td></tr>'
        table += '</tbody></table>'
        parts.append(table)

        return "\n".join(parts)

    def _comp_body(self, result: ComparisonResult) -> str:
        """Build comparison section body (reusable for combined report)."""
        parts: List[str] = []
        metrics = result._metric_table()

        if metrics:
            labels_js = json.dumps([m["label"] for m in metrics])
            workers_js = json.dumps([m["workers"] for m in metrics])
            tasks_js = json.dumps([m["tasks"] for m in metrics])

            parts.append('<div class="chart-container">')
            parts.append('<div class="chart-title">Workers &amp; Tasks</div>')
            parts.append('<canvas id="combCompChart"></canvas></div>')
            parts.append(f"""<script>
            drawBarChart('combCompChart', {labels_js}, [
                {{label: 'Workers', data: {workers_js}, color: '#58a6ff'}},
                {{label: 'Tasks', data: {tasks_js}, color: '#3fb950'}}
            ]);
            </script>""")

            table = '<table><thead><tr>'
            table += '<th>Scenario</th><th>Workers</th><th>Tasks</th>'
            table += '<th>Repl OK</th><th>Denied</th><th>Efficiency</th>'
            table += '</tr></thead><tbody>'
            for m in metrics:
                table += f'<tr><td><strong>{self._esc(m["label"])}</strong></td>'
                table += f'<td>{m["workers"]}</td><td>{m["tasks"]}</td>'
                table += f'<td>{m["repl_ok"]}</td><td>{m["repl_denied"]}</td>'
                table += f'<td>{m["efficiency"]:.2f}</td></tr>'
            table += '</tbody></table>'
            parts.append(table)

        return "\n".join(parts)

    # ─── File operations ────────────────────────────────────────────

    @staticmethod
    def save(html: str, path: str) -> str:
        """Save HTML report to a file. Returns the absolute path."""
        import os
        abs_path = os.path.abspath(path)
        os.makedirs(os.path.dirname(abs_path) or ".", exist_ok=True)
        with open(abs_path, "w", encoding="utf-8") as f:
            f.write(html)
        return abs_path


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════


def main() -> None:
    """CLI entry point for the HTML report generator."""
    import argparse
    import os
    import sys
    import webbrowser

    parser = argparse.ArgumentParser(
        description="AI Replication Sandbox — Interactive HTML Report Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Generate self-contained interactive HTML reports with charts and
visualizations from simulation, threat, or comparison analyses.

Examples:
  python -m replication.reporter --simulation                 # simulation report
  python -m replication.reporter --threats                    # threat assessment
  python -m replication.reporter --comparison                 # strategy comparison
  python -m replication.reporter --all                        # combined report
  python -m replication.reporter --all -o report.html         # save to specific file
  python -m replication.reporter --threats --open             # open in browser
  python -m replication.reporter --simulation --strategy burst --max-depth 4
        """,
    )

    parser.add_argument("--simulation", action="store_true", help="Include simulation report")
    parser.add_argument("--threats", action="store_true", help="Include threat assessment")
    parser.add_argument("--comparison", action="store_true", help="Include strategy comparison")
    parser.add_argument("--all", action="store_true", help="Include all report types")

    # Simulation config
    parser.add_argument("--strategy", choices=["greedy", "conservative", "random", "chain", "burst"],
                        default="greedy", help="Simulation strategy (default: greedy)")
    parser.add_argument("--scenario", choices=["minimal", "balanced", "stress", "chain", "burst"],
                        help="Use a built-in scenario preset")
    parser.add_argument("--max-depth", type=int, default=3, help="Max replication depth")
    parser.add_argument("--max-replicas", type=int, default=10, help="Max replicas")
    parser.add_argument("--seed", type=int, help="Random seed")

    # Output
    parser.add_argument("-o", "--output", type=str, help="Output file path (default: auto-generated)")
    parser.add_argument("--open", action="store_true", help="Open report in default browser")
    parser.add_argument("--stdout", action="store_true", help="Print HTML to stdout instead of file")

    args = parser.parse_args()

    # Default to --all if nothing specified
    if not (args.simulation or args.threats or args.comparison or args.all):
        args.all = True

    include_sim = args.simulation or args.all
    include_threats = args.threats or args.all
    include_comp = args.comparison or args.all

    reporter = HTMLReporter()

    # Build config
    if args.scenario:
        from .simulator import PRESETS
        config = PRESETS[args.scenario]
    else:
        config = ScenarioConfig(
            max_depth=args.max_depth,
            max_replicas=args.max_replicas,
            strategy=args.strategy,
            seed=args.seed,
            cooldown_seconds=0.0,
        )

    sim_report = None
    threat_report_obj = None
    comp_result = None

    if include_sim:
        from .simulator import Simulator
        sim = Simulator(config)
        sim_report = sim.run()
        print(f"✓ Simulation complete: {len(sim_report.workers)} workers, "
              f"{sim_report.total_tasks} tasks", file=sys.stderr)

    if include_threats:
        from .threats import ThreatSimulator, ThreatConfig
        threat_config = ThreatConfig(
            max_depth=config.max_depth,
            max_replicas=config.max_replicas,
        )
        tsim = ThreatSimulator(threat_config)
        threat_report_obj = tsim.run_all()
        print(f"✓ Threat analysis complete: score {threat_report_obj.security_score:.0f}/100 "
              f"({threat_report_obj.grade})", file=sys.stderr)

    if include_comp:
        from .comparator import Comparator
        comp = Comparator(base_config=config)
        comp_result = comp.compare_strategies(seed=args.seed)
        print(f"✓ Strategy comparison complete: {len(comp_result.runs)} strategies",
              file=sys.stderr)

    # Generate HTML
    report_count = sum(1 for x in [sim_report, threat_report_obj, comp_result] if x)
    if report_count > 1:
        html = reporter.combined_report(
            simulation=sim_report,
            threat=threat_report_obj,
            comparison=comp_result,
        )
    elif sim_report:
        html = reporter.simulation_report(sim_report)
    elif threat_report_obj:
        html = reporter.threat_report(threat_report_obj)
    elif comp_result:
        html = reporter.comparison_report(comp_result)
    else:
        print("No reports generated.", file=sys.stderr)
        sys.exit(1)

    if args.stdout:
        print(html)
    else:
        output_path = args.output or f"replication-report-{datetime.now().strftime('%Y%m%d-%H%M%S')}.html"
        saved_path = reporter.save(html, output_path)
        print(f"✓ Report saved to {saved_path}", file=sys.stderr)

        if args.open:
            webbrowser.open(f"file://{saved_path}")
            print(f"✓ Opened in browser", file=sys.stderr)


if __name__ == "__main__":
    main()
