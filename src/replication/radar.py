"""Safety Radar Chart — interactive radar visualization of scorecard results.

Generates a self-contained HTML page with a canvas-based radar chart
that visualizes safety scorecard dimension scores.  Supports:

* **Single scorecard** — radar polygon showing all dimension scores
* **Multi-scorecard overlay** — compare up to 5 configurations
* **Interactive hover** — dimension details on mouse-over
* **Import/export** — load scorecard JSON, download chart as PNG
* **Live demo** — built-in sample data for quick exploration

Usage (CLI)::

    python -m replication radar                          # generate with sample data
    python -m replication radar -o radar.html            # custom output path
    python -m replication radar --scenario greedy        # run scorecard + render
    python -m replication radar --json results.json      # from saved scorecard JSON

Programmatic::

    from replication.radar import generate_radar_html
    html = generate_radar_html()                         # sample data
    html = generate_radar_html(scorecard_results=[r1, r2])  # from results

"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


@dataclass
class RadarDataset:
    """A single dataset (scorecard result) for the radar chart."""
    label: str
    scores: Dict[str, float]  # dimension_name -> score (0-100)
    grades: Dict[str, str]    # dimension_name -> letter grade
    color: str = "#58a6ff"

    def to_dict(self) -> Dict[str, Any]:
        return {"label": self.label, "scores": self.scores, "grades": self.grades, "color": self.color}


SAMPLE_DATASETS: List[RadarDataset] = [
    RadarDataset(
        label="Standard Policy",
        scores={"Containment": 82, "Resource Control": 75, "Communication Safety": 88,
                "Behavioral Bounds": 70, "Kill Switch": 95, "Lineage Tracking": 78,
                "Threat Detection": 65, "Policy Compliance": 90},
        grades={"Containment": "B-", "Resource Control": "C", "Communication Safety": "B+",
                "Behavioral Bounds": "C-", "Kill Switch": "A", "Lineage Tracking": "C+",
                "Threat Detection": "D", "Policy Compliance": "A-"},
        color="#58a6ff",
    ),
    RadarDataset(
        label="Strict Policy",
        scores={"Containment": 95, "Resource Control": 92, "Communication Safety": 97,
                "Behavioral Bounds": 88, "Kill Switch": 98, "Lineage Tracking": 90,
                "Threat Detection": 85, "Policy Compliance": 96},
        grades={"Containment": "A", "Resource Control": "A-", "Communication Safety": "A+",
                "Behavioral Bounds": "B+", "Kill Switch": "A+", "Lineage Tracking": "A-",
                "Threat Detection": "B", "Policy Compliance": "A"},
        color="#3fb950",
    ),
    RadarDataset(
        label="Permissive Policy",
        scores={"Containment": 55, "Resource Control": 48, "Communication Safety": 62,
                "Behavioral Bounds": 45, "Kill Switch": 80, "Lineage Tracking": 50,
                "Threat Detection": 40, "Policy Compliance": 58},
        grades={"Containment": "F", "Resource Control": "F", "Communication Safety": "D-",
                "Behavioral Bounds": "F", "Kill Switch": "B-", "Lineage Tracking": "F",
                "Threat Detection": "F", "Policy Compliance": "F"},
        color="#f85149",
    ),
]

COLORS = ["#58a6ff", "#3fb950", "#f85149", "#d29922", "#bc8cff"]


def _html_template() -> str:
    return textwrap.dedent('''\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Safety Radar Chart — AI Replication Sandbox</title>
<style>
:root { --bg:#0d1117; --surface:#161b22; --border:#30363d; --text:#e6edf3; --muted:#8b949e; --accent:#58a6ff; }
* { margin:0; padding:0; box-sizing:border-box; }
body { font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif; background:var(--bg); color:var(--text); }
.header { padding:20px 32px; border-bottom:1px solid var(--border); display:flex; align-items:center; gap:16px; }
.header h1 { font-size:1.4em; }
.header .tag { background:var(--accent); color:#000; padding:2px 10px; border-radius:12px; font-size:0.75em; font-weight:600; }
.container { display:flex; gap:24px; padding:24px 32px; min-height:calc(100vh - 80px); }
.chart-area { flex:1; display:flex; flex-direction:column; align-items:center; }
.chart-area canvas { background:var(--surface); border:1px solid var(--border); border-radius:12px; }
.panel { width:340px; }
.card { background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:16px; margin-bottom:16px; }
.card h3 { color:var(--accent); font-size:0.95em; margin-bottom:10px; }
.legend-item { display:flex; align-items:center; gap:8px; padding:4px 0; font-size:0.85em; cursor:pointer; }
.legend-dot { width:14px; height:14px; border-radius:50%; flex-shrink:0; }
.legend-item.disabled { opacity:0.3; }
.score-table { width:100%; border-collapse:collapse; font-size:0.82em; }
.score-table th { text-align:left; color:var(--muted); padding:4px 6px; border-bottom:1px solid var(--border); }
.score-table td { padding:4px 6px; border-bottom:1px solid var(--border); }
.grade { font-weight:700; padding:1px 6px; border-radius:4px; font-size:0.85em; }
.grade-a { background:#3fb95033; color:#3fb950; }
.grade-b { background:#58a6ff33; color:#58a6ff; }
.grade-c { background:#d2992233; color:#d29922; }
.grade-d { background:#f0883e33; color:#f0883e; }
.grade-f { background:#f8514933; color:#f85149; }
.btn { background:var(--border); color:var(--text); border:none; padding:6px 14px; border-radius:6px; cursor:pointer; font-size:0.85em; margin:2px; }
.btn:hover { background:#30363dcc; }
.btn.primary { background:var(--accent); color:#000; }
.tooltip { position:fixed; background:var(--surface); border:1px solid var(--border); border-radius:8px; padding:10px 14px; font-size:0.82em; pointer-events:none; z-index:100; max-width:220px; display:none; }
.tooltip .dim-name { color:var(--accent); font-weight:600; margin-bottom:4px; }
.file-input { display:none; }
</style>
</head>
<body>
<div class="header">
    <h1>🎯 Safety Radar Chart</h1>
    <span class="tag">AI Replication Sandbox</span>
</div>
<div class="container">
    <div class="chart-area">
        <canvas id="radar" width="600" height="600"></canvas>
        <div style="margin-top:12px;display:flex;gap:8px;">
            <button class="btn" id="btnPNG">⬇ Download PNG</button>
            <button class="btn" id="btnImport">📂 Import JSON</button>
            <button class="btn" id="btnReset">↻ Reset</button>
            <input type="file" id="fileInput" class="file-input" accept=".json">
        </div>
    </div>
    <div class="panel">
        <div class="card">
            <h3>📊 Datasets</h3>
            <div id="legend"></div>
        </div>
        <div class="card">
            <h3>📋 Dimension Scores</h3>
            <select id="datasetSelect" style="width:100%;background:var(--bg);color:var(--text);border:1px solid var(--border);padding:4px 8px;border-radius:4px;margin-bottom:8px;font-size:0.85em;"></select>
            <table class="score-table">
                <thead><tr><th>Dimension</th><th>Score</th><th>Grade</th></tr></thead>
                <tbody id="scoreBody"></tbody>
            </table>
        </div>
        <div class="card">
            <h3>ℹ️ About</h3>
            <p style="font-size:0.82em;color:var(--muted);line-height:1.5;">
                Radar chart visualizing safety scorecard dimensions. Each axis represents a safety dimension (0-100).
                Click legend items to toggle datasets. Hover chart axes for details. Import scorecard JSON to visualize your own results.
            </p>
        </div>
    </div>
</div>
<div class="tooltip" id="tooltip"></div>

<script>
(function(){
'use strict';
const canvas = document.getElementById('radar');
const ctx = canvas.getContext('2d');
const W = 600, H = 600, CX = W/2, CY = H/2, R = 220;
const dpr = window.devicePixelRatio || 1;
canvas.width = W * dpr; canvas.height = H * dpr;
canvas.style.width = W + 'px'; canvas.style.height = H + 'px';
ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

let datasets = %%DATASETS%%;
let enabled = datasets.map(() => true);
let dimensions = Object.keys(datasets[0].scores);
const N = dimensions.length;

function angleFor(i) { return (i / N) * Math.PI * 2 - Math.PI / 2; }

function draw() {
    ctx.clearRect(0, 0, W, H);

    // Grid rings
    [0.2, 0.4, 0.6, 0.8, 1.0].forEach(f => {
        ctx.beginPath();
        for (let i = 0; i <= N; i++) {
            const a = angleFor(i % N);
            const x = CX + Math.cos(a) * R * f;
            const y = CY + Math.sin(a) * R * f;
            i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        }
        ctx.closePath();
        ctx.strokeStyle = '#30363d'; ctx.lineWidth = 1; ctx.stroke();
        // Ring label
        ctx.fillStyle = '#6e7681'; ctx.font = '10px sans-serif'; ctx.textAlign = 'right';
        ctx.fillText(String(Math.round(f * 100)), CX - 4, CY - R * f + 3);
    });

    // Axes
    for (let i = 0; i < N; i++) {
        const a = angleFor(i);
        ctx.beginPath(); ctx.moveTo(CX, CY);
        ctx.lineTo(CX + Math.cos(a) * R, CY + Math.sin(a) * R);
        ctx.strokeStyle = '#30363d55'; ctx.lineWidth = 1; ctx.stroke();

        // Labels
        const lx = CX + Math.cos(a) * (R + 24);
        const ly = CY + Math.sin(a) * (R + 24);
        ctx.fillStyle = '#e6edf3'; ctx.font = 'bold 12px sans-serif';
        ctx.textAlign = Math.abs(Math.cos(a)) < 0.01 ? 'center' : Math.cos(a) > 0 ? 'left' : 'right';
        ctx.textBaseline = Math.abs(Math.sin(a)) < 0.2 ? 'middle' : Math.sin(a) > 0 ? 'top' : 'bottom';
        ctx.fillText(dimensions[i], lx, ly);
    }

    // Dataset polygons
    datasets.forEach((ds, di) => {
        if (!enabled[di]) return;
        ctx.beginPath();
        for (let i = 0; i <= N; i++) {
            const a = angleFor(i % N);
            const dim = dimensions[i % N];
            const val = (ds.scores[dim] || 0) / 100;
            const x = CX + Math.cos(a) * R * val;
            const y = CY + Math.sin(a) * R * val;
            i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        }
        ctx.closePath();
        ctx.fillStyle = ds.color + '22'; ctx.fill();
        ctx.strokeStyle = ds.color; ctx.lineWidth = 2; ctx.stroke();

        // Data points
        for (let i = 0; i < N; i++) {
            const a = angleFor(i);
            const dim = dimensions[i];
            const val = (ds.scores[dim] || 0) / 100;
            const x = CX + Math.cos(a) * R * val;
            const y = CY + Math.sin(a) * R * val;
            ctx.beginPath(); ctx.arc(x, y, 4, 0, Math.PI * 2);
            ctx.fillStyle = ds.color; ctx.fill();
        }
    });
}

// Legend
function buildLegend() {
    const el = document.getElementById('legend');
    el.innerHTML = '';
    datasets.forEach((ds, i) => {
        const div = document.createElement('div');
        div.className = 'legend-item' + (enabled[i] ? '' : ' disabled');
        div.innerHTML = '<span class="legend-dot" style="background:' + ds.color + '"></span><span>' + ds.label + '</span>';
        div.onclick = () => { enabled[i] = !enabled[i]; div.classList.toggle('disabled'); draw(); };
        el.appendChild(div);
    });
}

// Score table
function buildScoreTable(idx) {
    const ds = datasets[idx];
    const body = document.getElementById('scoreBody');
    body.innerHTML = '';
    dimensions.forEach(dim => {
        const score = ds.scores[dim] || 0;
        const grade = ds.grades[dim] || '?';
        const gc = grade.startsWith('A') ? 'grade-a' : grade.startsWith('B') ? 'grade-b' : grade.startsWith('C') ? 'grade-c' : grade.startsWith('D') ? 'grade-d' : 'grade-f';
        const tr = document.createElement('tr');
        tr.innerHTML = '<td>' + dim + '</td><td>' + score.toFixed(0) + '</td><td><span class="grade ' + gc + '">' + grade + '</span></td>';
        body.appendChild(tr);
    });
}

function buildSelect() {
    const sel = document.getElementById('datasetSelect');
    sel.innerHTML = '';
    datasets.forEach((ds, i) => {
        const opt = document.createElement('option');
        opt.value = i; opt.textContent = ds.label;
        sel.appendChild(opt);
    });
    sel.onchange = () => buildScoreTable(+sel.value);
}

// Tooltip
const tooltip = document.getElementById('tooltip');
canvas.addEventListener('mousemove', e => {
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;

    let hit = -1;
    for (let i = 0; i < N; i++) {
        const a = angleFor(i);
        const lx = CX + Math.cos(a) * (R + 10);
        const ly = CY + Math.sin(a) * (R + 10);
        if (Math.abs(mx - lx) < 50 && Math.abs(my - ly) < 16) { hit = i; break; }
    }

    if (hit >= 0) {
        const dim = dimensions[hit];
        let html = '<div class="dim-name">' + dim + '</div>';
        datasets.forEach((ds, di) => {
            if (!enabled[di]) return;
            const s = ds.scores[dim] || 0;
            const g = ds.grades[dim] || '?';
            html += '<div style="color:' + ds.color + '">' + ds.label + ': ' + s.toFixed(0) + ' (' + g + ')</div>';
        });
        tooltip.innerHTML = html;
        tooltip.style.display = 'block';
        tooltip.style.left = (e.clientX + 12) + 'px';
        tooltip.style.top = (e.clientY + 12) + 'px';
    } else {
        tooltip.style.display = 'none';
    }
});
canvas.addEventListener('mouseleave', () => { tooltip.style.display = 'none'; });

// PNG export
document.getElementById('btnPNG').onclick = () => {
    const link = document.createElement('a');
    link.download = 'safety-radar.png';
    link.href = canvas.toDataURL('image/png');
    link.click();
};

// Import
document.getElementById('btnImport').onclick = () => document.getElementById('fileInput').click();
document.getElementById('fileInput').addEventListener('change', e => {
    const file = e.target.files[0]; if (!file) return;
    const reader = new FileReader();
    reader.onload = ev => {
        try {
            const data = JSON.parse(ev.target.result);
            // Support array of datasets or single scorecard result
            if (Array.isArray(data)) {
                data.forEach((d, i) => {
                    if (d.scores) datasets.push({ label: d.label || 'Import ' + (i+1), scores: d.scores, grades: d.grades || {}, color: ['#58a6ff','#3fb950','#f85149','#d29922','#bc8cff'][datasets.length % 5] });
                });
            } else if (data.dimensions) {
                // Scorecard result format
                const scores = {}, grades = {};
                data.dimensions.forEach(d => { scores[d.name] = d.score; grades[d.name] = d.grade; });
                datasets.push({ label: data.overall_grade ? 'Scorecard (' + data.overall_grade + ')' : 'Imported', scores, grades, color: ['#58a6ff','#3fb950','#f85149','#d29922','#bc8cff'][datasets.length % 5] });
            } else if (data.scores) {
                datasets.push({ label: data.label || 'Imported', scores: data.scores, grades: data.grades || {}, color: ['#58a6ff','#3fb950','#f85149','#d29922','#bc8cff'][datasets.length % 5] });
            }
            dimensions = [...new Set(datasets.flatMap(d => Object.keys(d.scores)))];
            enabled = datasets.map(() => true);
            buildLegend(); buildSelect(); buildScoreTable(0); draw();
        } catch(err) { alert('Import error: ' + err.message); }
    };
    reader.readAsText(file);
    e.target.value = '';
});

// Reset
document.getElementById('btnReset').onclick = () => {
    datasets = %%DATASETS%%;
    dimensions = Object.keys(datasets[0].scores);
    enabled = datasets.map(() => true);
    buildLegend(); buildSelect(); buildScoreTable(0); draw();
};

buildLegend(); buildSelect(); buildScoreTable(0); draw();
})();
</script>
</body>
</html>
''')


def generate_radar_html(
    scorecard_results: Optional[List[Any]] = None,
    datasets: Optional[List[RadarDataset]] = None,
) -> str:
    """Generate self-contained radar chart HTML.

    Args:
        scorecard_results: list of ScorecardResult objects (from scorecard module)
        datasets: list of RadarDataset objects (manual)

    Returns:
        Complete HTML string
    """
    if datasets:
        ds_list = [d.to_dict() for d in datasets]
    elif scorecard_results:
        ds_list = []
        for i, result in enumerate(scorecard_results):
            scores = {}
            grades = {}
            for dim in result.dimensions:
                scores[dim.name] = dim.score
                grades[dim.name] = dim.grade
            ds_list.append({
                "label": f"{result.overall_grade} — {result.config.strategy}",
                "scores": scores,
                "grades": grades,
                "color": COLORS[i % len(COLORS)],
            })
    else:
        ds_list = [d.to_dict() for d in SAMPLE_DATASETS]

    html = _html_template()
    return html.replace("%%DATASETS%%", json.dumps(ds_list))


def main(argv: Optional[list] = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="replication radar",
        description="Generate interactive safety radar chart",
    )
    parser.add_argument("-o", "--output", default="radar.html",
                        help="Output HTML file (default: radar.html)")
    parser.add_argument("--scenario", choices=["greedy", "conservative", "balanced", "random", "adversarial"],
                        help="Run scorecard with this scenario and render results")
    parser.add_argument("--json", metavar="FILE",
                        help="Load scorecard JSON results from file")
    args = parser.parse_args(argv)

    datasets = None
    scorecard_results = None

    if args.json:
        with open(args.json) as f:
            data = json.load(f)
        if isinstance(data, list):
            datasets = [
                RadarDataset(
                    label=d.get("label", f"Config {i+1}"),
                    scores=d["scores"],
                    grades=d.get("grades", {}),
                    color=COLORS[i % len(COLORS)],
                )
                for i, d in enumerate(data)
            ]
        elif "dimensions" in data:
            scores = {d["name"]: d["score"] for d in data["dimensions"]}
            grades = {d["name"]: d["grade"] for d in data["dimensions"]}
            datasets = [RadarDataset(label=data.get("overall_grade", "Result"), scores=scores, grades=grades)]
        print(f"Loaded {len(datasets or [])} dataset(s) from {args.json}")

    elif args.scenario:
        from .scorecard import SafetyScorecard, ScorecardConfig
        from .simulator import ScenarioConfig, PRESETS
        print(f"Running scorecard with scenario '{args.scenario}'...")
        sc = SafetyScorecard()
        preset = PRESETS.get(args.scenario, PRESETS["balanced"])
        config = ScenarioConfig(**{k: v for k, v in preset.items() if k != "name"})
        config.strategy = args.scenario
        result = sc.evaluate(config, ScorecardConfig(quick=True))
        scorecard_results = [result]
        print(f"Scorecard complete: {result.overall_grade} ({result.overall_score:.0f}/100)")

    html = generate_radar_html(scorecard_results=scorecard_results, datasets=datasets)
    out_path = os.path.abspath(args.output)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    size_kb = len(html) / 1024
    print(f"Radar chart saved to {out_path} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
