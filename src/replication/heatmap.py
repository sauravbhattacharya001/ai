"""Safety Heatmap — 2D parameter sweep with interactive HTML visualization.

Sweeps two parameters simultaneously across a grid of values, running
simulations at each intersection to produce a safety-score heatmap.
Generates a self-contained HTML file with an interactive heatmap
(color-coded cells, hover tooltips, click-to-inspect), axis labels,
and configurable color scales.

Usage (CLI)::

    python -m replication.heatmap --x max_depth --y max_replicas
    python -m replication.heatmap --x max_depth --y cooldown_seconds --runs 10
    python -m replication.heatmap --x max_depth --y max_replicas --output heatmap.html
    python -m replication.heatmap --x max_depth --y max_replicas --json
    python -m replication.heatmap --x max_depth --y max_replicas --metric threat_score
    python -m replication.heatmap --list-params
    python -m replication.heatmap --list-metrics

Programmatic::

    from replication.heatmap import SafetyHeatmap, HeatmapConfig
    hm = SafetyHeatmap(HeatmapConfig(x_param="max_depth", y_param="max_replicas"))
    result = hm.sweep()
    result.to_html("heatmap.html")
    print(result.render())
"""

from __future__ import annotations

import argparse
import html as html_mod
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from .simulator import Simulator, ScenarioConfig, SimulationReport, PRESETS
from .sensitivity import PARAM_DEFS, ParameterDef
from ._helpers import box_header as _box_header, stats_mean as _mean, stats_std as _std


# ── Metrics ──────────────────────────────────────────────────────────

@dataclass
class CellMetrics:
    """Aggregated metrics for one grid cell."""

    x_value: Any
    y_value: Any
    safety_score: float
    threat_score: float
    breach_rate: float
    avg_depth: float
    avg_replicas: float
    avg_duration: float
    containment_rate: float
    raw_reports: List[SimulationReport] = field(default_factory=list, repr=False)


METRIC_DEFS: Dict[str, Tuple[str, str, bool]] = {
    "safety_score": ("Safety Score", "Overall safety score 0-100", True),
    "threat_score": ("Threat Score", "Composite threat level 0-100", False),
    "breach_rate": ("Breach Rate", "Fraction of runs with safety breaches", False),
    "containment_rate": ("Containment Rate", "Fraction of runs successfully contained", True),
    "avg_depth": ("Avg Depth", "Average replication depth reached", False),
    "avg_replicas": ("Avg Replicas", "Average peak replica count", False),
    "avg_duration": ("Avg Duration", "Average simulation duration in seconds", False),
}


@dataclass
class HeatmapConfig:
    """Configuration for a 2D parameter sweep."""

    x_param: str = "max_depth"
    y_param: str = "max_replicas"
    x_values: Optional[List[Any]] = None
    y_values: Optional[List[Any]] = None
    runs_per_cell: int = 5
    metric: str = "safety_score"
    strategy: str = "greedy"
    title: Optional[str] = None
    color_scale: str = "safety"


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _safety_color(value: float, vmin: float, vmax: float) -> str:
    """Green for high, red for low."""
    if vmax == vmin:
        return "#4caf50"
    t = (value - vmin) / (vmax - vmin)
    if t < 0.5:
        r, g = 220, int(_lerp(50, 200, t * 2))
    else:
        r, g = int(_lerp(220, 60, (t - 0.5) * 2)), 180
    return f"#{r:02x}{g:02x}50"


def _threat_color(value: float, vmin: float, vmax: float) -> str:
    """Red for high, green for low."""
    if vmax == vmin:
        return "#4caf50"
    t = (value - vmin) / (vmax - vmin)
    if t < 0.5:
        r, g = int(_lerp(60, 220, t * 2)), 180
    else:
        r, g = 220, int(_lerp(200, 50, (t - 0.5) * 2))
    return f"#{r:02x}{g:02x}50"


def _extract_metrics(reports: List[SimulationReport]) -> Dict[str, float]:
    """Extract averaged metrics from simulation reports."""
    if not reports:
        return {
            "safety_score": 0.0, "threat_score": 100.0, "breach_rate": 1.0,
            "avg_depth": 0.0, "avg_replicas": 0.0, "avg_duration": 0.0,
            "containment_rate": 0.0,
        }

    safety_scores: List[float] = []
    depths: List[float] = []
    replicas: List[float] = []
    durations: List[float] = []
    breaches = 0
    contained = 0

    for r in reports:
        violations = len(r.violations) if hasattr(r, "violations") else 0
        max_depth = r.config.max_depth if hasattr(r, "config") else 5
        reached_depth = r.max_depth_reached if hasattr(r, "max_depth_reached") else 0
        peak = r.peak_workers if hasattr(r, "peak_workers") else 0
        quota = r.config.max_replicas if hasattr(r, "config") else 10

        depth_ratio = reached_depth / max(max_depth, 1)
        replica_ratio = peak / max(quota, 1)
        violation_penalty = min(violations * 10, 50)
        score = max(0.0, 100 - depth_ratio * 25 - replica_ratio * 25 - violation_penalty)
        safety_scores.append(score)
        depths.append(float(reached_depth))
        replicas.append(float(peak))
        durations.append(r.elapsed if hasattr(r, "elapsed") else 0.0)
        if violations > 0:
            breaches += 1
        if hasattr(r, "contained") and r.contained:
            contained += 1

    n = len(reports)
    return {
        "safety_score": _mean(safety_scores),
        "threat_score": max(0.0, 100 - _mean(safety_scores)),
        "breach_rate": breaches / n,
        "avg_depth": _mean(depths),
        "avg_replicas": _mean(replicas),
        "avg_duration": _mean(durations),
        "containment_rate": contained / n,
    }


@dataclass
class HeatmapResult:
    """Result of a 2D parameter sweep."""

    config: HeatmapConfig
    grid: List[List[CellMetrics]]
    x_values: List[Any]
    y_values: List[Any]
    elapsed: float = 0.0
    total_simulations: int = 0

    def get_metric_grid(self, metric: Optional[str] = None) -> List[List[float]]:
        m = metric or self.config.metric
        return [[getattr(cell, m) for cell in row] for row in self.grid]

    def get_bounds(self, metric: Optional[str] = None) -> Tuple[float, float]:
        flat = [v for row in self.get_metric_grid(metric) for v in row]
        return (min(flat), max(flat)) if flat else (0.0, 0.0)

    def best_cell(self, metric: Optional[str] = None) -> CellMetrics:
        m = metric or self.config.metric
        _, _, higher_better = METRIC_DEFS.get(m, (m, m, True))
        best = None
        best_val = None
        for row in self.grid:
            for cell in row:
                v = getattr(cell, m)
                if best_val is None or (higher_better and v > best_val) or (not higher_better and v < best_val):
                    best, best_val = cell, v
        return best  # type: ignore[return-value]

    def worst_cell(self, metric: Optional[str] = None) -> CellMetrics:
        m = metric or self.config.metric
        _, _, higher_better = METRIC_DEFS.get(m, (m, m, True))
        worst = None
        worst_val = None
        for row in self.grid:
            for cell in row:
                v = getattr(cell, m)
                if worst_val is None or (higher_better and v < worst_val) or (not higher_better and v > worst_val):
                    worst, worst_val = cell, v
        return worst  # type: ignore[return-value]

    def render(self) -> str:
        m = self.config.metric
        disp, _, _ = METRIC_DEFS.get(m, (m, m, True))
        x_def = PARAM_DEFS.get(self.config.x_param)
        y_def = PARAM_DEFS.get(self.config.y_param)
        x_name = x_def.display_name if x_def else self.config.x_param
        y_name = y_def.display_name if y_def else self.config.y_param

        lines = list(_box_header(f"Safety Heatmap: {disp}"))
        lines.append(f"  X-axis: {x_name}  |  Y-axis: {y_name}")
        lines.append(f"  Runs/cell: {self.config.runs_per_cell}  |  "
                      f"Total sims: {self.total_simulations}  |  "
                      f"Elapsed: {self.elapsed:.1f}s")
        lines.append("")

        col_w = 8
        header = f"{'':>{col_w}}"
        for xv in self.x_values:
            header += f"{str(xv):>{col_w}}"
        lines.append(header)
        lines.append("  " + "\u2500" * (col_w * (len(self.x_values) + 1)))

        for yi, row in enumerate(self.grid):
            label = f"{self.y_values[yi]:>{col_w}}"
            cells = ""
            for cell in row:
                v = getattr(cell, m)
                cells += f"{v:>{col_w}.1f}"
            lines.append(f"{label}{cells}")

        lines.append("")
        best = self.best_cell()
        worst = self.worst_cell()
        lines.append(f"  Best:  {x_name}={best.x_value}, {y_name}={best.y_value} "
                      f"\u2192 {disp}={getattr(best, m):.1f}")
        lines.append(f"  Worst: {x_name}={worst.x_value}, {y_name}={worst.y_value} "
                      f"\u2192 {disp}={getattr(worst, m):.1f}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": {
                "x_param": self.config.x_param, "y_param": self.config.y_param,
                "runs_per_cell": self.config.runs_per_cell,
                "metric": self.config.metric, "strategy": self.config.strategy,
            },
            "x_values": self.x_values, "y_values": self.y_values,
            "metric": self.config.metric,
            "grid": [[{
                "x": c.x_value, "y": c.y_value,
                "safety_score": round(c.safety_score, 2),
                "threat_score": round(c.threat_score, 2),
                "breach_rate": round(c.breach_rate, 3),
                "containment_rate": round(c.containment_rate, 3),
                "avg_depth": round(c.avg_depth, 2),
                "avg_replicas": round(c.avg_replicas, 2),
                "avg_duration": round(c.avg_duration, 3),
            } for c in row] for row in self.grid],
            "elapsed": round(self.elapsed, 2),
            "total_simulations": self.total_simulations,
            "generated": datetime.now(timezone.utc).isoformat(),
        }

    def to_html(self, path: Optional[str] = None) -> str:
        m = self.config.metric
        disp, desc, higher_better = METRIC_DEFS.get(m, (m, m, True))
        x_def = PARAM_DEFS.get(self.config.x_param)
        y_def = PARAM_DEFS.get(self.config.y_param)
        x_name = x_def.display_name if x_def else self.config.x_param
        y_name = y_def.display_name if y_def else self.config.y_param
        vmin, vmax = self.get_bounds()
        color_fn = _safety_color if higher_better else _threat_color

        cells_html = []
        for yi, row in enumerate(self.grid):
            for xi, cell in enumerate(row):
                val = getattr(cell, m)
                bg = color_fn(val, vmin, vmax)
                tooltip = (
                    f"{x_name}={cell.x_value}, {y_name}={cell.y_value}\\n"
                    f"Safety: {cell.safety_score:.1f}  Threat: {cell.threat_score:.1f}\\n"
                    f"Breach: {cell.breach_rate:.0%}  Contained: {cell.containment_rate:.0%}\\n"
                    f"Depth: {cell.avg_depth:.1f}  Replicas: {cell.avg_replicas:.1f}"
                )
                cells_html.append(
                    f'<div class="cell" style="grid-row:{yi+2};grid-column:{xi+2};'
                    f'background:{bg}" title="{tooltip}" '
                    f'data-x="{cell.x_value}" data-y="{cell.y_value}" '
                    f'data-val="{val:.1f}">{val:.1f}</div>'
                )

        x_labels = "".join(
            f'<div class="xlabel" style="grid-row:1;grid-column:{i+2}">{v}</div>'
            for i, v in enumerate(self.x_values))
        y_labels = "".join(
            f'<div class="ylabel" style="grid-row:{i+2};grid-column:1">{v}</div>'
            for i, v in enumerate(self.y_values))

        ncols, nrows = len(self.x_values), len(self.y_values)
        best, worst = self.best_cell(), self.worst_cell()
        title = self.config.title or f"Safety Heatmap: {disp}"

        page = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>{html_mod.escape(title)}</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:system-ui,sans-serif;background:#1a1a2e;color:#e0e0e0;display:flex;flex-direction:column;align-items:center;padding:2rem}}
h1{{font-size:1.5rem;margin-bottom:.5rem;color:#fff}}
.sub{{color:#aaa;margin-bottom:1.5rem;font-size:.9rem}}
.grid{{display:grid;grid-template-columns:80px repeat({ncols},60px);grid-template-rows:40px repeat({nrows},60px);gap:2px;margin-bottom:1.5rem}}
.cell{{display:flex;align-items:center;justify-content:center;font-size:.8rem;font-weight:600;color:#000;border-radius:4px;cursor:pointer;transition:transform .15s,box-shadow .15s}}
.cell:hover{{transform:scale(1.15);box-shadow:0 0 12px rgba(255,255,255,.4);z-index:10}}
.cell.sel{{outline:3px solid #fff;z-index:11}}
.xlabel,.ylabel{{display:flex;align-items:center;justify-content:center;font-size:.75rem;color:#ccc;font-weight:500}}
.ax{{font-size:.85rem;color:#aaa;margin-bottom:.5rem}}
.info{{background:#16213e;border-radius:8px;padding:1rem 1.5rem;max-width:600px;width:100%;margin-top:1rem}}
.info h3{{margin-bottom:.5rem;color:#e94560}}
.info table{{width:100%;border-collapse:collapse}}
.info td{{padding:4px 8px;border-bottom:1px solid #1a1a3e}}
.info td:first-child{{color:#aaa;width:45%}}
.leg{{display:flex;align-items:center;gap:.5rem;margin-top:1rem;font-size:.8rem}}
.leg-bar{{width:200px;height:16px;border-radius:4px;background:linear-gradient(90deg,#dc3250,#dcc850,#3cb450)}}
.stats{{display:flex;gap:2rem;margin-top:1rem;font-size:.85rem}}
.stats span{{color:#aaa}}
</style></head><body>
<h1>{html_mod.escape(title)}</h1>
<div class="sub">{html_mod.escape(desc)} &mdash; {x_name} &times; {y_name} ({self.config.runs_per_cell} runs/cell, {self.total_simulations} total)</div>
<div class="ax">&larr; {html_mod.escape(x_name)} &rarr;</div>
<div class="grid">{x_labels}{y_labels}{"".join(cells_html)}</div>
<div class="ax">&uarr; {html_mod.escape(y_name)}</div>
<div class="leg"><span>{"Unsafe" if higher_better else "Safe"}</span><div class="leg-bar"></div><span>{"Safe" if higher_better else "Unsafe"}</span></div>
<div class="stats">
<div><span>Best:</span> {x_name}={best.x_value}, {y_name}={best.y_value} &rarr; {getattr(best, m):.1f}</div>
<div><span>Worst:</span> {x_name}={worst.x_value}, {y_name}={worst.y_value} &rarr; {getattr(worst, m):.1f}</div>
</div>
<div class="info" id="d"><h3>Click a cell to inspect</h3><p>Hover for quick stats, click for full detail.</p></div>
<script>
document.querySelectorAll('.cell').forEach(el=>{{
el.addEventListener('click',()=>{{
document.querySelectorAll('.cell.sel').forEach(s=>s.classList.remove('sel'));
el.classList.add('sel');
const d=document.getElementById('d'),t=el.title.replace(/\\\\n/g,'\\n').split('\\n');
d.innerHTML='<h3>'+t[0]+'</h3><table>'+t.slice(1).map(l=>{{
const p=l.split(/\\s{{2,}}/);return p.map(x=>'<tr><td>'+x.split(': ')[0]+'</td><td>'+(x.split(': ')[1]||'')+'</td></tr>').join('');
}}).join('')+'</table>';
}});}});
</script></body></html>"""

        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(page)
        return page


class SafetyHeatmap:
    """2D parameter sweep engine for safety analysis."""

    def __init__(self, config: Optional[HeatmapConfig] = None) -> None:
        self.config = config or HeatmapConfig()

    def sweep(self) -> HeatmapResult:
        cfg = self.config
        if cfg.x_param == cfg.y_param:
            raise ValueError(f"x_param and y_param must differ, got '{cfg.x_param}' for both")
        if cfg.x_param not in PARAM_DEFS:
            raise ValueError(f"Unknown parameter '{cfg.x_param}'. Available: {', '.join(sorted(PARAM_DEFS))}")
        if cfg.y_param not in PARAM_DEFS:
            raise ValueError(f"Unknown parameter '{cfg.y_param}'. Available: {', '.join(sorted(PARAM_DEFS))}")
        if cfg.metric not in METRIC_DEFS:
            raise ValueError(f"Unknown metric '{cfg.metric}'. Available: {', '.join(sorted(METRIC_DEFS))}")

        x_def, y_def = PARAM_DEFS[cfg.x_param], PARAM_DEFS[cfg.y_param]
        x_vals = cfg.x_values or x_def.values
        y_vals = cfg.y_values or y_def.values
        strategy = cfg.strategy  # ScenarioConfig.strategy is a str

        t0 = time.monotonic()
        total_sims = 0
        grid: List[List[CellMetrics]] = []

        for yv in y_vals:
            row: List[CellMetrics] = []
            for xv in x_vals:
                base = ScenarioConfig(strategy=strategy)
                x_def.setter(base, xv)
                y_def.setter(base, yv)
                reports = [Simulator(base).run() for _ in range(cfg.runs_per_cell)]
                total_sims += len(reports)
                metrics = _extract_metrics(reports)
                row.append(CellMetrics(
                    x_value=xv, y_value=yv,
                    safety_score=metrics["safety_score"],
                    threat_score=metrics["threat_score"],
                    breach_rate=metrics["breach_rate"],
                    avg_depth=metrics["avg_depth"],
                    avg_replicas=metrics["avg_replicas"],
                    avg_duration=metrics["avg_duration"],
                    containment_rate=metrics["containment_rate"],
                    raw_reports=reports,
                ))
            grid.append(row)

        return HeatmapResult(
            config=cfg, grid=grid, x_values=x_vals, y_values=y_vals,
            elapsed=time.monotonic() - t0, total_simulations=total_sims,
        )


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(prog="replication.heatmap",
                                  description="2D parameter sweep \u2192 interactive safety heatmap")
    ap.add_argument("--x", dest="x_param", default="max_depth")
    ap.add_argument("--y", dest="y_param", default="max_replicas")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--metric", default="safety_score", choices=sorted(METRIC_DEFS))
    ap.add_argument("--strategy", default="greedy")
    ap.add_argument("--output", "-o")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--list-params", action="store_true")
    ap.add_argument("--list-metrics", action="store_true")
    ap.add_argument("--title")
    args = ap.parse_args(argv)

    if args.list_params:
        print("Sweepable parameters:")
        for n, p in sorted(PARAM_DEFS.items()):
            print(f"  {n:20s} {p.display_name} \u2014 {p.description}")
            print(f"  {'':20s} values: {p.values}")
        return
    if args.list_metrics:
        print("Available metrics:")
        for n, (d, desc, hib) in sorted(METRIC_DEFS.items()):
            print(f"  {n:20s} {d} \u2014 {desc} ({'\u2191 higher=better' if hib else '\u2193 lower=better'})")
        return

    cfg = HeatmapConfig(x_param=args.x_param, y_param=args.y_param,
                         runs_per_cell=args.runs, metric=args.metric,
                         strategy=args.strategy, title=args.title)
    result = SafetyHeatmap(cfg).sweep()
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(result.render())
    if args.output:
        result.to_html(args.output)
        print(f"\n  \u2713 HTML heatmap written to {args.output}")


if __name__ == "__main__":
    main()
