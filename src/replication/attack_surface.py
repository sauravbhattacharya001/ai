"""attack_surface — Interactive HTML attack surface visualizer.

Generates a self-contained HTML page that renders the replication
sandbox's attack trees as an interactive sunburst diagram.  Each
ring represents a depth level of the attack tree; nodes are
color-coded by risk level (difficulty × likelihood).

Features:

- **Sunburst layout**: hierarchical radial chart — click to zoom
  into sub-trees, click center to zoom out.
- **Risk heatmap coloring**: green (low risk) → red (critical risk),
  with mitigated nodes shown in cool blue.
- **Hover details**: cost, likelihood, difficulty, mitigation status.
- **Goal selector**: switch between the 5 threat goals.
- **Coverage ring**: animated progress ring showing mitigation coverage.
- **Summary stats**: total paths, avg risk, cheapest path, leaf count.
- **Search**: filter nodes by label text.
- **Export**: save current view as PNG.

Usage (CLI)::

    python -m replication attack-surface
    python -m replication attack-surface -o surface.html
    python -m replication attack-surface --open

Usage (API)::

    from replication.attack_surface import generate_surface
    html = generate_surface()
    Path("attack_surface.html").write_text(html)
"""

from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from pathlib import Path
from typing import Any, Dict, List, Optional

from .attack_tree import (
    AttackNode,
    AttackTreeGenerator,
    AttackTreeResult,
    Difficulty,
    NodeType,
    TreeAnalysis,
    TreeConfig,
)


def _node_to_sunburst(node: AttackNode) -> Dict[str, Any]:
    """Convert an AttackNode tree into a nested dict for the sunburst."""
    risk = node.likelihood * (1.0 - (node.difficulty.numeric - 1) / 4)
    entry: Dict[str, Any] = {
        "name": node.label,
        "id": node.id,
        "type": node.node_type.value,
        "risk": round(risk, 3),
        "mitigated": node.mitigated,
    }
    if node.is_leaf():
        entry["value"] = 1
        entry["cost"] = node.cost
        entry["difficulty"] = node.difficulty.value
        entry["likelihood"] = node.likelihood
        if node.mitigation:
            entry["mitigation"] = node.mitigation
    if node.children:
        entry["children"] = [_node_to_sunburst(c) for c in node.children]
    return entry


def _analysis_to_json(analysis: TreeAnalysis) -> Dict[str, Any]:
    """Convert a TreeAnalysis to a JSON-serializable summary."""
    return {
        "goal": analysis.goal.value,
        "tree": _node_to_sunburst(analysis.root),
        "leafCount": analysis.leaf_count,
        "mitigatedCount": analysis.mitigated_count,
        "coveragePct": round(
            100 * analysis.mitigated_count / max(1, analysis.leaf_count), 1
        ),
        "overallRisk": round(analysis.overall_risk, 1),
        "pathCount": len(analysis.all_paths),
        "cheapestPath": (
            analysis.min_cost_path.to_dict()
            if analysis.min_cost_path
            else None
        ),
        "highestLikelihoodPath": (
            analysis.max_likelihood_path.to_dict()
            if analysis.max_likelihood_path
            else None
        ),
    }


def generate_surface(
    config: Optional[TreeConfig] = None,
    result: Optional[AttackTreeResult] = None,
) -> str:
    """Generate a self-contained HTML attack surface visualization.

    Parameters
    ----------
    config : TreeConfig, optional
        Attack tree configuration.  Uses defaults if omitted.
    result : AttackTreeResult, optional
        Pre-computed analysis result.  If omitted, ``analyze()``
        is called with *config*.

    Returns
    -------
    str
        Complete HTML page as a string.
    """
    if result is None:
        cfg = config or TreeConfig()
        gen = AttackTreeGenerator(cfg)
        result = gen.analyze()

    data = [_analysis_to_json(a) for a in result.analyses]
    data_json = json.dumps(data, indent=None)

    return _HTML_TEMPLATE.replace("/* __DATA__ */", f"const DATA = {data_json};")


# ── HTML template ────────────────────────────────────────────────────

_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Attack Surface Visualizer — AI Replication Safety</title>
<style>
:root {
  --bg: #0d1117; --surface: #161b22; --border: #30363d;
  --text: #e6edf3; --muted: #8b949e; --accent: #58a6ff;
  --risk-low: #2ea043; --risk-med: #d29922; --risk-high: #f85149;
  --mitigated: #388bfd;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
  background: var(--bg); color: var(--text);
  min-height: 100vh; display: flex; flex-direction: column;
}
header {
  padding: 16px 24px; border-bottom: 1px solid var(--border);
  display: flex; align-items: center; gap: 16px; flex-wrap: wrap;
}
header h1 { font-size: 18px; font-weight: 600; }
header h1 span { color: var(--accent); }
.controls { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
select, input[type=text] {
  background: var(--surface); color: var(--text); border: 1px solid var(--border);
  padding: 6px 10px; border-radius: 6px; font-size: 13px;
}
select:focus, input:focus { outline: none; border-color: var(--accent); }
.main { display: flex; flex: 1; overflow: hidden; }
.viz { flex: 1; display: flex; justify-content: center; align-items: center; position: relative; }
.sidebar {
  width: 320px; border-left: 1px solid var(--border); padding: 16px;
  overflow-y: auto; flex-shrink: 0;
}
.sidebar h2 { font-size: 14px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 12px; }
.stat { display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid var(--border); }
.stat-label { color: var(--muted); font-size: 13px; }
.stat-value { font-weight: 600; font-size: 14px; }
.coverage-ring {
  width: 100px; height: 100px; margin: 16px auto;
  position: relative; display: flex; align-items: center; justify-content: center;
}
.coverage-ring svg { transform: rotate(-90deg); }
.coverage-ring .pct {
  position: absolute; font-size: 20px; font-weight: 700;
}
.path-card {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 8px; padding: 12px; margin-top: 12px; font-size: 13px;
}
.path-card h3 { font-size: 13px; color: var(--accent); margin-bottom: 6px; }
.path-step { color: var(--muted); padding: 2px 0; }
.path-step::before { content: "→ "; color: var(--border); }
.legend { display: flex; gap: 12px; flex-wrap: wrap; margin-top: 16px; }
.legend-item { display: flex; align-items: center; gap: 4px; font-size: 12px; color: var(--muted); }
.legend-dot { width: 10px; height: 10px; border-radius: 50%; }
.tooltip {
  position: fixed; background: var(--surface); border: 1px solid var(--border);
  border-radius: 8px; padding: 10px 14px; pointer-events: none;
  font-size: 12px; z-index: 100; max-width: 280px; display: none;
  box-shadow: 0 4px 12px rgba(0,0,0,0.5);
}
.tooltip .tt-label { font-weight: 600; margin-bottom: 4px; }
.tooltip .tt-row { color: var(--muted); padding: 1px 0; }
.breadcrumb {
  position: absolute; top: 12px; left: 12px; font-size: 12px;
  color: var(--muted); cursor: pointer;
}
.breadcrumb:hover { color: var(--accent); }
canvas { cursor: pointer; }
.btn {
  background: var(--surface); color: var(--text); border: 1px solid var(--border);
  padding: 6px 12px; border-radius: 6px; cursor: pointer; font-size: 12px;
}
.btn:hover { border-color: var(--accent); }
@media (max-width: 768px) {
  .main { flex-direction: column; }
  .sidebar { width: 100%; border-left: none; border-top: 1px solid var(--border); max-height: 300px; }
}
</style>
</head>
<body>
<header>
  <h1>🛡️ Attack Surface <span>Visualizer</span></h1>
  <div class="controls">
    <select id="goalSelect"></select>
    <input type="text" id="search" placeholder="Search nodes…" style="width:160px">
    <button class="btn" id="exportBtn">📷 Export PNG</button>
  </div>
</header>
<div class="main">
  <div class="viz">
    <span class="breadcrumb" id="breadcrumb"></span>
    <canvas id="canvas" width="700" height="700"></canvas>
  </div>
  <div class="sidebar" id="sidebar"></div>
</div>
<div class="tooltip" id="tooltip"></div>
<script>
/* __DATA__ */

const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const tooltip = document.getElementById('tooltip');
const goalSelect = document.getElementById('goalSelect');
const searchInput = document.getElementById('search');
const sidebar = document.getElementById('sidebar');
const breadcrumb = document.getElementById('breadcrumb');

let currentGoalIdx = 0;
let zoomRoot = null;
let arcs = []; // painted arc metadata for hit testing
let searchTerm = '';

// Populate goal selector
DATA.forEach((d, i) => {
  const opt = document.createElement('option');
  opt.value = i;
  opt.textContent = d.goal.replace(/_/g, ' ').toUpperCase();
  goalSelect.appendChild(opt);
});
goalSelect.addEventListener('change', () => {
  currentGoalIdx = +goalSelect.value;
  zoomRoot = null;
  render();
});
searchInput.addEventListener('input', () => {
  searchTerm = searchInput.value.toLowerCase();
  render();
});

// Risk → color
function riskColor(risk, mitigated) {
  if (mitigated) return 'rgba(56,139,253,0.7)';
  if (risk < 0.3) return 'rgba(46,160,67,0.7)';
  if (risk < 0.6) return 'rgba(210,153,34,0.7)';
  return 'rgba(248,81,73,0.7)';
}
function riskColorSolid(risk, mitigated) {
  if (mitigated) return '#388bfd';
  if (risk < 0.3) return '#2ea043';
  if (risk < 0.6) return '#d29922';
  return '#f85149';
}

// Flatten tree to get all leaf nodes with their angles
function flattenTree(node) {
  if (!node.children || node.children.length === 0) return [node];
  let leaves = [];
  for (const c of node.children) leaves = leaves.concat(flattenTree(c));
  return leaves;
}

function countLeaves(node) {
  if (!node.children || node.children.length === 0) return 1;
  return node.children.reduce((s, c) => s + countLeaves(c), 0);
}

// Assign angular extents
function layoutSunburst(root, startAngle, endAngle, depth) {
  const totalLeaves = countLeaves(root);
  const result = [];
  const highlight = searchTerm && root.name.toLowerCase().includes(searchTerm);

  result.push({
    node: root, depth, startAngle, endAngle, highlight,
  });

  if (root.children && root.children.length > 0) {
    let angle = startAngle;
    for (const child of root.children) {
      const childLeaves = countLeaves(child);
      const childEnd = angle + (endAngle - startAngle) * (childLeaves / totalLeaves);
      result.push(...layoutSunburst(child, angle, childEnd, depth + 1));
      angle = childEnd;
    }
  }
  return result;
}

function render() {
  const data = DATA[currentGoalIdx];
  const root = zoomRoot || data.tree;
  const W = canvas.width, H = canvas.height;
  const cx = W / 2, cy = H / 2;
  const maxR = Math.min(cx, cy) - 30;
  const levels = maxDepth(root);
  const ringW = maxR / (levels + 1);
  const innerR = ringW * 0.6;

  ctx.clearRect(0, 0, W, H);
  arcs = [];

  // Layout
  const layout = layoutSunburst(root, 0, Math.PI * 2, 0);

  // Draw arcs
  for (const item of layout) {
    const r1 = innerR + item.depth * ringW;
    const r2 = r1 + ringW - 2;
    if (r1 > maxR) continue;

    const color = riskColor(item.node.risk, item.node.mitigated);
    ctx.beginPath();
    ctx.arc(cx, cy, r2, item.startAngle, item.endAngle);
    ctx.arc(cx, cy, r1, item.endAngle, item.startAngle, true);
    ctx.closePath();
    ctx.fillStyle = item.highlight ? '#fff' : color;
    ctx.globalAlpha = item.highlight ? 0.95 : 0.85;
    ctx.fill();
    ctx.globalAlpha = 1;
    ctx.strokeStyle = '#0d1117';
    ctx.lineWidth = 1;
    ctx.stroke();

    arcs.push({ ...item, r1, r2, cx, cy });

    // Label for wide-enough arcs
    const arcLen = (item.endAngle - item.startAngle) * (r1 + r2) / 2;
    if (arcLen > 40 && ringW > 18) {
      const midAngle = (item.startAngle + item.endAngle) / 2;
      const labelR = (r1 + r2) / 2;
      const tx = cx + Math.cos(midAngle) * labelR;
      const ty = cy + Math.sin(midAngle) * labelR;
      ctx.save();
      ctx.translate(tx, ty);
      let rot = midAngle;
      if (rot > Math.PI / 2 && rot < Math.PI * 1.5) rot += Math.PI;
      ctx.rotate(rot);
      ctx.fillStyle = '#e6edf3';
      ctx.font = '10px sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      const label = item.node.name.length > 18 ? item.node.name.slice(0, 16) + '…' : item.node.name;
      ctx.fillText(label, 0, 0);
      ctx.restore();
    }
  }

  // Center circle — goal name
  ctx.beginPath();
  ctx.arc(cx, cy, innerR, 0, Math.PI * 2);
  ctx.fillStyle = '#161b22';
  ctx.fill();
  ctx.strokeStyle = '#30363d';
  ctx.lineWidth = 2;
  ctx.stroke();
  ctx.fillStyle = '#e6edf3';
  ctx.font = 'bold 11px sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  const goalLabel = root.name.length > 22 ? root.name.slice(0, 20) + '…' : root.name;
  ctx.fillText(goalLabel, cx, cy - 6);
  ctx.font = '10px sans-serif';
  ctx.fillStyle = '#8b949e';
  ctx.fillText('click to zoom', cx, cy + 10);

  // Breadcrumb
  if (zoomRoot) {
    breadcrumb.textContent = '← Back to root';
    breadcrumb.style.display = 'block';
  } else {
    breadcrumb.style.display = 'none';
  }

  // Sidebar
  renderSidebar(data);
}

function maxDepth(node) {
  if (!node.children || node.children.length === 0) return 0;
  return 1 + Math.max(...node.children.map(maxDepth));
}

function renderSidebar(data) {
  const cov = data.coveragePct;
  const dashOffset = 251.3 * (1 - cov / 100);

  let html = `
    <h2>📊 Summary</h2>
    <div class="stat"><span class="stat-label">Overall Risk</span><span class="stat-value" style="color:${riskColorSolid(data.overallRisk/100, false)}">${data.overallRisk}</span></div>
    <div class="stat"><span class="stat-label">Attack Paths</span><span class="stat-value">${data.pathCount}</span></div>
    <div class="stat"><span class="stat-label">Leaf Nodes</span><span class="stat-value">${data.leafCount}</span></div>
    <div class="stat"><span class="stat-label">Mitigated</span><span class="stat-value">${data.mitigatedCount} / ${data.leafCount}</span></div>

    <h2 style="margin-top:20px">🛡️ Mitigation Coverage</h2>
    <div class="coverage-ring">
      <svg width="100" height="100" viewBox="0 0 100 100">
        <circle cx="50" cy="50" r="40" fill="none" stroke="#30363d" stroke-width="8"/>
        <circle cx="50" cy="50" r="40" fill="none" stroke="${cov > 70 ? '#2ea043' : cov > 40 ? '#d29922' : '#f85149'}"
                stroke-width="8" stroke-dasharray="251.3" stroke-dashoffset="${dashOffset}"
                stroke-linecap="round"/>
      </svg>
      <span class="pct">${cov}%</span>
    </div>
  `;

  if (data.cheapestPath) {
    html += `
    <h2 style="margin-top:20px">💰 Cheapest Attack Path</h2>
    <div class="path-card">
      <h3>Cost: ${data.cheapestPath.total_cost} · Likelihood: ${(data.cheapestPath.combined_likelihood * 100).toFixed(1)}%</h3>
      ${data.cheapestPath.path.map(s => `<div class="path-step">${s}</div>`).join('')}
    </div>`;
  }

  if (data.highestLikelihoodPath) {
    html += `
    <h2 style="margin-top:16px">⚡ Highest Likelihood Path</h2>
    <div class="path-card">
      <h3>Likelihood: ${(data.highestLikelihoodPath.combined_likelihood * 100).toFixed(1)}% · Cost: ${data.highestLikelihoodPath.total_cost}</h3>
      ${data.highestLikelihoodPath.path.map(s => `<div class="path-step">${s}</div>`).join('')}
    </div>`;
  }

  html += `
    <div class="legend">
      <div class="legend-item"><span class="legend-dot" style="background:#2ea043"></span>Low Risk</div>
      <div class="legend-item"><span class="legend-dot" style="background:#d29922"></span>Medium</div>
      <div class="legend-item"><span class="legend-dot" style="background:#f85149"></span>High Risk</div>
      <div class="legend-item"><span class="legend-dot" style="background:#388bfd"></span>Mitigated</div>
    </div>
  `;

  sidebar.innerHTML = html;
}

// Hit test
function hitTest(mx, my) {
  for (let i = arcs.length - 1; i >= 0; i--) {
    const a = arcs[i];
    const dx = mx - a.cx, dy = my - a.cy;
    const dist = Math.sqrt(dx * dx + dy * dy);
    if (dist < a.r1 || dist > a.r2) continue;
    let angle = Math.atan2(dy, dx);
    if (angle < 0) angle += Math.PI * 2;
    if (angle >= a.startAngle && angle <= a.endAngle) return a;
  }
  return null;
}

// Tooltip
canvas.addEventListener('mousemove', (e) => {
  const rect = canvas.getBoundingClientRect();
  const mx = e.clientX - rect.left;
  const my = e.clientY - rect.top;
  const hit = hitTest(mx, my);
  if (hit) {
    const n = hit.node;
    let html = `<div class="tt-label">${n.name}</div>`;
    html += `<div class="tt-row">Type: ${n.type}</div>`;
    html += `<div class="tt-row">Risk: ${(n.risk * 100).toFixed(0)}%</div>`;
    if (n.difficulty) html += `<div class="tt-row">Difficulty: ${n.difficulty}</div>`;
    if (n.likelihood !== undefined) html += `<div class="tt-row">Likelihood: ${(n.likelihood * 100).toFixed(0)}%</div>`;
    if (n.cost !== undefined) html += `<div class="tt-row">Cost: ${n.cost}</div>`;
    if (n.mitigated) html += `<div class="tt-row" style="color:#388bfd">✓ Mitigated${n.mitigation ? ': ' + n.mitigation : ''}</div>`;
    tooltip.innerHTML = html;
    tooltip.style.display = 'block';
    tooltip.style.left = (e.clientX + 12) + 'px';
    tooltip.style.top = (e.clientY + 12) + 'px';
  } else {
    tooltip.style.display = 'none';
  }
});
canvas.addEventListener('mouseleave', () => { tooltip.style.display = 'none'; });

// Click to zoom
canvas.addEventListener('click', (e) => {
  const rect = canvas.getBoundingClientRect();
  const mx = e.clientX - rect.left;
  const my = e.clientY - rect.top;
  const cx = canvas.width / 2, cy = canvas.height / 2;

  // Click center → zoom out
  const dist = Math.sqrt((mx - cx) ** 2 + (my - cy) ** 2);
  if (dist < 30) {
    zoomRoot = null;
    render();
    return;
  }

  const hit = hitTest(mx, my);
  if (hit && hit.node.children && hit.node.children.length > 0) {
    zoomRoot = hit.node;
    render();
  }
});

// Breadcrumb click
breadcrumb.addEventListener('click', () => { zoomRoot = null; render(); });

// Export PNG
document.getElementById('exportBtn').addEventListener('click', () => {
  const link = document.createElement('a');
  link.download = `attack-surface-${DATA[currentGoalIdx].goal}.png`;
  link.href = canvas.toDataURL('image/png');
  link.click();
});

// DPI scaling
function resizeCanvas() {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.parentElement.getBoundingClientRect();
  const size = Math.min(rect.width - 40, rect.height - 40, 700);
  canvas.style.width = size + 'px';
  canvas.style.height = size + 'px';
  canvas.width = size * dpr;
  canvas.height = size * dpr;
  ctx.scale(dpr, dpr);
  // Reset scale vars for hit testing
  canvas._scale = dpr;
  canvas._size = size;
  render();
}

// Override hit test coords for DPI
const origHitTest = hitTest;

window.addEventListener('resize', resizeCanvas);
resizeCanvas();
</script>
</body>
</html>
"""


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for attack surface visualization."""
    parser = argparse.ArgumentParser(
        prog="python -m replication attack-surface",
        description="Generate an interactive attack surface visualization (HTML)",
    )
    parser.add_argument(
        "-o", "--output",
        default="attack_surface.html",
        help="Output HTML file path (default: attack_surface.html)",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open the generated page in the default browser",
    )
    args = parser.parse_args(argv)

    config = TreeConfig()

    html = generate_surface(config=config)
    out = Path(args.output)
    out.write_text(html, encoding="utf-8")
    print(f"Attack surface visualization written to {out}")

    if args.open:
        webbrowser.open(str(out.resolve()))


if __name__ == "__main__":
    main()
