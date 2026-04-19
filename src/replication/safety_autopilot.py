"""Safety Autopilot — autonomous safety monitoring loop.

A goal-directed safety monitor that continuously evaluates agent fleet health,
detects degradation patterns, and takes corrective actions with configurable
human-in-the-loop gates.  Think of it as cruise-control for safety: set a
target safety posture and the autopilot keeps the system on track.

Key capabilities:

- **Continuous monitoring loop** — tick-based evaluation cycle with
  configurable interval that checks multiple safety dimensions each tick
- **Goal-directed behaviour** — define a target safety profile (min score,
  max incidents, max drift) and autopilot steers toward it
- **Escalation ladder** — 4-level response (log → alert → recommend → auto-act)
  with configurable human-approval gates before destructive actions
- **Action playbook** — built-in corrective actions (tighten thresholds,
  quarantine workers, rotate credentials, trigger drills) with dry-run mode
- **Flight recorder** — append-only log of every observation, decision, and
  action for full auditability
- **Situational awareness** — correlates signals across scorecard, drift,
  incidents, and capacity to produce a single "flight status"
- **Demo mode** — simulated fleet with injected anomalies to showcase the
  autopilot in action

Usage::

    python -m replication autopilot --demo
    python -m replication autopilot --ticks 20 --interval 1.0
    python -m replication autopilot --dry-run --target-score 80
    python -m replication autopilot --export json
    python -m replication autopilot --export html -o autopilot_report.html

"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# ── enums & constants ────────────────────────────────────────────────

class Severity(Enum):
    OK = "ok"
    ADVISORY = "advisory"
    WARNING = "warning"
    CRITICAL = "critical"


class ActionType(Enum):
    LOG = "log"
    ALERT = "alert"
    RECOMMEND = "recommend"
    AUTO_ACT = "auto_act"


class CorrectiveAction(Enum):
    TIGHTEN_THRESHOLDS = "tighten_thresholds"
    QUARANTINE_WORKER = "quarantine_worker"
    ROTATE_CREDENTIALS = "rotate_credentials"
    TRIGGER_DRILL = "trigger_drill"
    SCALE_MONITORS = "scale_monitors"
    PAUSE_REPLICATION = "pause_replication"
    NOTIFY_OPERATOR = "notify_operator"


FLIGHT_STATUS_LABELS = {
    Severity.OK: "✅ CRUISING",
    Severity.ADVISORY: "🟡 ADVISORY",
    Severity.WARNING: "🟠 TURBULENCE",
    Severity.CRITICAL: "🔴 MAYDAY",
}


# ── data structures ──────────────────────────────────────────────────

@dataclass
class SafetyGoal:
    """Target safety posture the autopilot steers toward."""
    min_score: float = 75.0
    max_incidents_per_tick: int = 2
    max_drift_pct: float = 15.0
    max_response_time_ms: float = 500.0
    min_capacity_headroom_pct: float = 20.0


@dataclass
class Observation:
    """Single point-in-time safety observation."""
    tick: int
    timestamp: str
    safety_score: float
    drift_pct: float
    incident_count: int
    capacity_used_pct: float
    response_time_ms: float
    anomaly_flags: List[str] = field(default_factory=list)


@dataclass
class Decision:
    """Autopilot decision with rationale."""
    tick: int
    timestamp: str
    severity: str
    rationale: str
    actions: List[str]
    approved: bool
    dry_run: bool


@dataclass
class FlightRecord:
    """Complete autopilot session log."""
    session_id: str
    started: str
    goal: Dict[str, Any]
    observations: List[Dict[str, Any]] = field(default_factory=list)
    decisions: List[Dict[str, Any]] = field(default_factory=list)
    actions_taken: List[Dict[str, Any]] = field(default_factory=list)
    summary: Optional[Dict[str, Any]] = None


# ── escalation engine ────────────────────────────────────────────────

# Map severity to allowed action levels
ESCALATION_LADDER: Dict[Severity, List[ActionType]] = {
    Severity.OK: [ActionType.LOG],
    Severity.ADVISORY: [ActionType.LOG, ActionType.ALERT],
    Severity.WARNING: [ActionType.LOG, ActionType.ALERT, ActionType.RECOMMEND],
    Severity.CRITICAL: [ActionType.LOG, ActionType.ALERT, ActionType.RECOMMEND, ActionType.AUTO_ACT],
}

# Map anomaly patterns to corrective actions
ACTION_PLAYBOOK: Dict[str, List[CorrectiveAction]] = {
    "score_drop": [CorrectiveAction.TIGHTEN_THRESHOLDS, CorrectiveAction.NOTIFY_OPERATOR],
    "drift_spike": [CorrectiveAction.QUARANTINE_WORKER, CorrectiveAction.TIGHTEN_THRESHOLDS],
    "incident_surge": [CorrectiveAction.PAUSE_REPLICATION, CorrectiveAction.ROTATE_CREDENTIALS],
    "capacity_crunch": [CorrectiveAction.SCALE_MONITORS, CorrectiveAction.NOTIFY_OPERATOR],
    "slow_response": [CorrectiveAction.SCALE_MONITORS, CorrectiveAction.TRIGGER_DRILL],
    "multi_signal": [CorrectiveAction.PAUSE_REPLICATION, CorrectiveAction.NOTIFY_OPERATOR,
                     CorrectiveAction.ROTATE_CREDENTIALS],
}


def assess_severity(obs: Observation, goal: SafetyGoal) -> Tuple[Severity, List[str]]:
    """Evaluate observation against goal, return severity + anomaly flags."""
    flags: List[str] = []

    if obs.safety_score < goal.min_score * 0.7:
        flags.append("score_drop")
    if obs.drift_pct > goal.max_drift_pct * 1.5:
        flags.append("drift_spike")
    if obs.incident_count > goal.max_incidents_per_tick * 2:
        flags.append("incident_surge")
    if obs.capacity_used_pct > (100 - goal.min_capacity_headroom_pct):
        flags.append("capacity_crunch")
    if obs.response_time_ms > goal.max_response_time_ms * 1.5:
        flags.append("slow_response")

    if len(flags) >= 3:
        flags.append("multi_signal")

    if len(flags) >= 3:
        return Severity.CRITICAL, flags
    elif len(flags) == 2:
        return Severity.WARNING, flags
    elif len(flags) == 1:
        return Severity.ADVISORY, flags
    return Severity.OK, flags


def select_actions(severity: Severity, flags: List[str]) -> List[CorrectiveAction]:
    """Select corrective actions based on severity and anomaly flags."""
    if severity == Severity.OK:
        return []

    actions: List[CorrectiveAction] = []
    seen = set()
    for f in flags:
        for action in ACTION_PLAYBOOK.get(f, []):
            if action not in seen:
                seen.add(action)
                actions.append(action)

    # Gate destructive actions behind severity level
    allowed = ESCALATION_LADDER[severity]
    if ActionType.AUTO_ACT not in allowed:
        # Filter out destructive actions — only notify/recommend
        non_destructive = {CorrectiveAction.NOTIFY_OPERATOR, CorrectiveAction.TIGHTEN_THRESHOLDS,
                           CorrectiveAction.SCALE_MONITORS, CorrectiveAction.TRIGGER_DRILL}
        actions = [a for a in actions if a in non_destructive]

    return actions


# ── simulated fleet for demo ─────────────────────────────────────────

class SimulatedFleet:
    """Simulated fleet that generates semi-realistic safety telemetry."""

    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)
        self._base_score = 85.0
        self._base_drift = 5.0
        self._tick = 0
        # Inject anomaly windows
        self._anomaly_windows = [
            (5, 8, "gradual_degradation"),
            (12, 14, "incident_spike"),
            (18, 20, "capacity_pressure"),
        ]

    def _in_anomaly(self) -> Optional[str]:
        for start, end, kind in self._anomaly_windows:
            if start <= self._tick <= end:
                return kind
        return None

    def observe(self) -> Observation:
        self._tick += 1
        anomaly = self._in_anomaly()

        # Base values with noise
        score = self._base_score + self._rng.gauss(0, 3)
        drift = self._base_drift + abs(self._rng.gauss(0, 2))
        incidents = max(0, int(self._rng.gauss(0.5, 0.8)))
        capacity = 55.0 + self._rng.gauss(0, 5)
        response = 200.0 + abs(self._rng.gauss(0, 50))

        # Inject anomalies
        if anomaly == "gradual_degradation":
            progress = (self._tick - 5) / 3.0
            score -= 15 * progress + self._rng.gauss(0, 3)
            drift += 10 * progress
        elif anomaly == "incident_spike":
            incidents += self._rng.randint(3, 7)
            score -= self._rng.uniform(5, 15)
            response += self._rng.uniform(100, 300)
        elif anomaly == "capacity_pressure":
            capacity += 25 + self._rng.uniform(0, 10)
            response += self._rng.uniform(50, 200)

        # Clamp
        score = max(0, min(100, score))
        drift = max(0, min(100, drift))
        capacity = max(0, min(100, capacity))
        response = max(10, response)

        return Observation(
            tick=self._tick,
            timestamp=datetime.now(timezone.utc).isoformat(),
            safety_score=round(score, 1),
            drift_pct=round(drift, 1),
            incident_count=incidents,
            capacity_used_pct=round(capacity, 1),
            response_time_ms=round(response, 1),
        )


# ── autopilot engine ─────────────────────────────────────────────────

class SafetyAutopilot:
    """Autonomous safety monitoring loop."""

    def __init__(
        self,
        goal: Optional[SafetyGoal] = None,
        dry_run: bool = False,
        auto_approve: bool = False,
        verbose: bool = True,
    ):
        self.goal = goal or SafetyGoal()
        self.dry_run = dry_run
        self.auto_approve = auto_approve
        self.verbose = verbose
        self._session_id = f"autopilot-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        self.flight_record = FlightRecord(
            session_id=self._session_id,
            started=datetime.now(timezone.utc).isoformat(),
            goal=asdict(self.goal),
        )
        self._score_history: List[float] = []

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def process_tick(self, obs: Observation) -> Decision:
        """Process one observation tick through the autopilot pipeline."""
        self.flight_record.observations.append(asdict(obs))
        self._score_history.append(obs.safety_score)

        # Assess
        severity, flags = assess_severity(obs, self.goal)
        obs.anomaly_flags = flags

        # Select actions
        actions = select_actions(severity, flags)

        # Trend analysis — detect sustained degradation
        rationale_parts = []
        if flags:
            rationale_parts.append(f"Anomalies detected: {', '.join(flags)}")
        if len(self._score_history) >= 3:
            recent = self._score_history[-3:]
            if all(recent[i] > recent[i + 1] for i in range(len(recent) - 1)):
                rationale_parts.append("Sustained score decline over 3 ticks")
                if severity.value == "ok":
                    severity = Severity.ADVISORY
                    actions = [CorrectiveAction.NOTIFY_OPERATOR]
        if not rationale_parts:
            rationale_parts.append("All metrics within target envelope")

        # Decide approval
        approved = True
        if actions and not self.auto_approve and not self.dry_run:
            # In non-interactive mode, auto-approve non-destructive
            destructive = {CorrectiveAction.QUARANTINE_WORKER,
                           CorrectiveAction.PAUSE_REPLICATION,
                           CorrectiveAction.ROTATE_CREDENTIALS}
            if any(a in destructive for a in actions):
                approved = False  # needs human approval

        decision = Decision(
            tick=obs.tick,
            timestamp=datetime.now(timezone.utc).isoformat(),
            severity=severity.value,
            rationale="; ".join(rationale_parts),
            actions=[a.value for a in actions],
            approved=approved,
            dry_run=self.dry_run,
        )
        self.flight_record.decisions.append(asdict(decision))

        # Execute actions
        status = FLIGHT_STATUS_LABELS[severity]
        self._log(f"\n  Tick {obs.tick:>3} │ {status} │ Score {obs.safety_score:5.1f} │ "
                   f"Drift {obs.drift_pct:4.1f}% │ Incidents {obs.incident_count} │ "
                   f"Capacity {obs.capacity_used_pct:4.1f}%")

        if actions:
            action_strs = [a.value for a in actions]
            if self.dry_run:
                self._log(f"         │ [DRY RUN] Would execute: {', '.join(action_strs)}")
            elif approved:
                self._log(f"         │ ⚡ Executing: {', '.join(action_strs)}")
                for a in actions:
                    self.flight_record.actions_taken.append({
                        "tick": obs.tick,
                        "action": a.value,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    })
            else:
                self._log(f"         │ 🔒 Pending human approval: {', '.join(action_strs)}")

        if flags:
            self._log(f"         │ Flags: {', '.join(flags)}")

        return decision

    def summarize(self) -> Dict[str, Any]:
        """Generate session summary."""
        observations = self.flight_record.observations
        if not observations:
            return {"status": "no data"}

        scores = [o["safety_score"] for o in observations]
        incidents = sum(o["incident_count"] for o in observations)
        severities = [d["severity"] for d in self.flight_record.decisions]

        summary = {
            "session_id": self._session_id,
            "total_ticks": len(observations),
            "score_min": min(scores),
            "score_max": max(scores),
            "score_avg": round(sum(scores) / len(scores), 1),
            "score_final": scores[-1],
            "total_incidents": incidents,
            "total_actions_taken": len(self.flight_record.actions_taken),
            "severity_counts": {s: severities.count(s) for s in set(severities)},
            "goal_met": scores[-1] >= self.goal.min_score,
            "dry_run": self.dry_run,
        }
        self.flight_record.summary = summary
        return summary


# ── output formatters ────────────────────────────────────────────────

def _format_text(record: FlightRecord) -> str:
    """Format flight record as readable text."""
    lines = [
        "=" * 72,
        f"  SAFETY AUTOPILOT — FLIGHT RECORD",
        f"  Session: {record.session_id}",
        f"  Started: {record.started}",
        "=" * 72,
        "",
        f"  Goal: score≥{record.goal['min_score']} | "
        f"drift≤{record.goal['max_drift_pct']}% | "
        f"incidents≤{record.goal['max_incidents_per_tick']}/tick | "
        f"headroom≥{record.goal['min_capacity_headroom_pct']}%",
        "",
    ]

    if record.summary:
        s = record.summary
        lines.append("  ── Summary ──")
        lines.append(f"  Ticks: {s['total_ticks']} | "
                      f"Score: {s['score_min']}–{s['score_max']} (avg {s['score_avg']}) | "
                      f"Incidents: {s['total_incidents']} | "
                      f"Actions: {s['total_actions_taken']}")
        lines.append(f"  Severities: {s['severity_counts']}")
        lines.append(f"  Goal met: {'✅ Yes' if s['goal_met'] else '❌ No'}")
        if s.get("dry_run"):
            lines.append("  Mode: DRY RUN (no actions executed)")
        lines.append("")

    if record.actions_taken:
        lines.append("  ── Actions Taken ──")
        for a in record.actions_taken:
            lines.append(f"  Tick {a['tick']:>3}: {a['action']}")
        lines.append("")

    lines.append("=" * 72)
    return "\n".join(lines)


def _format_json(record: FlightRecord) -> str:
    return json.dumps(asdict(record), indent=2)


def _generate_html(record: FlightRecord) -> str:
    """Generate interactive HTML autopilot report."""
    obs_json = json.dumps(record.observations)
    dec_json = json.dumps(record.decisions)
    summary = record.summary or {}
    actions_json = json.dumps(record.actions_taken)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Safety Autopilot — Flight Record</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: #0a0e17; color: #e0e0e0; padding: 24px; }}
h1 {{ color: #60a5fa; margin-bottom: 8px; font-size: 1.6em; }}
.meta {{ color: #888; font-size: 0.85em; margin-bottom: 24px; }}
.cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
          gap: 12px; margin-bottom: 24px; }}
.card {{ background: #151b2e; border-radius: 10px; padding: 16px; text-align: center; }}
.card .val {{ font-size: 1.8em; font-weight: 700; }}
.card .lbl {{ font-size: 0.75em; color: #888; margin-top: 4px; }}
.ok {{ color: #4ade80; }} .advisory {{ color: #facc15; }}
.warning {{ color: #fb923c; }} .critical {{ color: #f87171; }}
canvas {{ width: 100%; height: 300px; background: #151b2e; border-radius: 10px;
          margin-bottom: 24px; }}
.log {{ background: #151b2e; border-radius: 10px; padding: 16px;
        max-height: 400px; overflow-y: auto; font-family: monospace; font-size: 0.82em; }}
.log-entry {{ padding: 4px 0; border-bottom: 1px solid #1e2740; }}
.severity-badge {{ padding: 2px 8px; border-radius: 4px; font-size: 0.75em; font-weight: 600; }}
.badge-ok {{ background: #16533050; color: #4ade80; }}
.badge-advisory {{ background: #71410050; color: #facc15; }}
.badge-warning {{ background: #7c290050; color: #fb923c; }}
.badge-critical {{ background: #7f1d1d50; color: #f87171; }}
</style>
</head>
<body>
<h1>🛩️ Safety Autopilot — Flight Record</h1>
<p class="meta">Session: {record.session_id} | Started: {record.started}
   | Ticks: {summary.get('total_ticks', 0)}
   {' | DRY RUN' if summary.get('dry_run') else ''}</p>

<div class="cards">
  <div class="card"><div class="val {'ok' if summary.get('goal_met') else 'critical'}">{summary.get('score_avg', '—')}</div><div class="lbl">Avg Score</div></div>
  <div class="card"><div class="val">{summary.get('score_min', '—')}–{summary.get('score_max', '—')}</div><div class="lbl">Score Range</div></div>
  <div class="card"><div class="val warning">{summary.get('total_incidents', 0)}</div><div class="lbl">Total Incidents</div></div>
  <div class="card"><div class="val advisory">{summary.get('total_actions_taken', 0)}</div><div class="lbl">Actions Taken</div></div>
  <div class="card"><div class="val {'ok' if summary.get('goal_met') else 'critical'}">{'✅' if summary.get('goal_met') else '❌'}</div><div class="lbl">Goal Met</div></div>
</div>

<canvas id="chart"></canvas>

<h2 style="margin-bottom:12px; font-size:1.1em;">📋 Decision Log</h2>
<div class="log" id="log"></div>

<script>
const obs = {obs_json};
const dec = {dec_json};
const actions = {actions_json};

// Chart
const canvas = document.getElementById('chart');
const ctx = canvas.getContext('2d');
function drawChart() {{
  const W = canvas.width = canvas.offsetWidth * 2;
  const H = canvas.height = canvas.offsetHeight * 2;
  ctx.scale(2, 2);
  const w = W / 2, h = H / 2;
  const pad = {{ t: 30, r: 20, b: 30, l: 50 }};
  const pw = w - pad.l - pad.r, ph = h - pad.t - pad.b;
  if (!obs.length) return;

  ctx.clearRect(0, 0, w, h);

  // Grid
  ctx.strokeStyle = '#1e2740'; ctx.lineWidth = 0.5;
  for (let y = 0; y <= 100; y += 25) {{
    const py = pad.t + ph * (1 - y / 100);
    ctx.beginPath(); ctx.moveTo(pad.l, py); ctx.lineTo(pad.l + pw, py); ctx.stroke();
    ctx.fillStyle = '#666'; ctx.font = '10px sans-serif'; ctx.textAlign = 'right';
    ctx.fillText(y, pad.l - 6, py + 3);
  }}

  // Goal line
  const goalY = pad.t + ph * (1 - {summary.get('score_avg', 75)} > 0 ? (1 - {record.goal['min_score']} / 100) : 0.25);
  ctx.strokeStyle = '#4ade8050'; ctx.lineWidth = 1; ctx.setLineDash([4, 4]);
  ctx.beginPath(); ctx.moveTo(pad.l, goalY); ctx.lineTo(pad.l + pw, goalY); ctx.stroke();
  ctx.setLineDash([]);

  // Score line
  ctx.strokeStyle = '#60a5fa'; ctx.lineWidth = 2;
  ctx.beginPath();
  obs.forEach((o, i) => {{
    const x = pad.l + (pw * i / (obs.length - 1 || 1));
    const y = pad.t + ph * (1 - o.safety_score / 100);
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }});
  ctx.stroke();

  // Drift line
  ctx.strokeStyle = '#fb923c80'; ctx.lineWidth = 1.5;
  ctx.beginPath();
  obs.forEach((o, i) => {{
    const x = pad.l + (pw * i / (obs.length - 1 || 1));
    const y = pad.t + ph * (1 - o.drift_pct / 100);
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }});
  ctx.stroke();

  // Incident bars
  const maxInc = Math.max(1, ...obs.map(o => o.incident_count));
  obs.forEach((o, i) => {{
    if (o.incident_count === 0) return;
    const x = pad.l + (pw * i / (obs.length - 1 || 1)) - 3;
    const bh = (o.incident_count / maxInc) * ph * 0.3;
    ctx.fillStyle = '#f8717140';
    ctx.fillRect(x, pad.t + ph - bh, 6, bh);
  }});

  // Legend
  ctx.font = '10px sans-serif';
  const leg = [['#60a5fa', 'Score'], ['#fb923c80', 'Drift'], ['#f8717140', 'Incidents']];
  let lx = pad.l;
  leg.forEach(([c, l]) => {{
    ctx.fillStyle = c; ctx.fillRect(lx, 8, 16, 8);
    ctx.fillStyle = '#aaa'; ctx.textAlign = 'left'; ctx.fillText(l, lx + 20, 16);
    lx += 80;
  }});
}}
drawChart();
window.addEventListener('resize', drawChart);

// Log
const logEl = document.getElementById('log');
dec.forEach(d => {{
  const badge = d.severity;
  const acts = d.actions.length ? ' → ' + d.actions.join(', ') : '';
  const dry = d.dry_run ? ' [DRY RUN]' : '';
  const approval = d.actions.length && !d.approved ? ' 🔒 PENDING' : '';
  logEl.innerHTML += '<div class="log-entry">'
    + '<span class="severity-badge badge-' + badge + '">' + badge.toUpperCase() + '</span> '
    + 'Tick ' + d.tick + ': ' + d.rationale + acts + dry + approval
    + '</div>';
}});
</script>
</body>
</html>"""


# ── CLI ──────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point for Safety Autopilot."""
    parser = argparse.ArgumentParser(
        prog="python -m replication autopilot",
        description="Safety Autopilot — autonomous safety monitoring loop",
    )
    parser.add_argument("--ticks", type=int, default=25,
                        help="Number of monitoring ticks (default: 25)")
    parser.add_argument("--interval", type=float, default=0.0,
                        help="Seconds between ticks in demo (default: 0)")
    parser.add_argument("--target-score", type=float, default=75.0,
                        help="Minimum acceptable safety score (default: 75)")
    parser.add_argument("--max-drift", type=float, default=15.0,
                        help="Maximum acceptable drift %% (default: 15)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show actions without executing")
    parser.add_argument("--auto-approve", action="store_true",
                        help="Auto-approve all corrective actions")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for demo fleet (default: 42)")
    parser.add_argument("--export", choices=["text", "json", "html"],
                        default="text", help="Output format")
    parser.add_argument("-o", "--output", help="Write output to file")
    parser.add_argument("--demo", action="store_true",
                        help="Run demo with simulated fleet anomalies")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="Suppress tick-by-tick output")
    args = parser.parse_args(argv)

    if args.demo:
        args.dry_run = True  # demo is always dry-run for safety

    goal = SafetyGoal(
        min_score=args.target_score,
        max_drift_pct=args.max_drift,
    )
    autopilot = SafetyAutopilot(
        goal=goal,
        dry_run=args.dry_run,
        auto_approve=args.auto_approve,
        verbose=not args.quiet,
    )

    fleet = SimulatedFleet(seed=args.seed)

    if not args.quiet:
        mode = "DRY RUN" if args.dry_run else "LIVE"
        print(f"\n🛩️  Safety Autopilot — {mode}")
        print(f"   Goal: score≥{goal.min_score} | drift≤{goal.max_drift_pct}% | "
              f"incidents≤{goal.max_incidents_per_tick}/tick")
        print(f"   Ticks: {args.ticks} | Seed: {args.seed}")
        print("─" * 72)

    for _ in range(args.ticks):
        obs = fleet.observe()
        autopilot.process_tick(obs)
        if args.interval > 0:
            time.sleep(args.interval)

    summary = autopilot.summarize()

    if not args.quiet:
        print("\n" + "─" * 72)
        print(f"  Session complete: {summary['total_ticks']} ticks | "
              f"Score {summary['score_min']}–{summary['score_max']} (avg {summary['score_avg']}) | "
              f"Incidents: {summary['total_incidents']} | "
              f"Actions: {summary['total_actions_taken']}")
        print(f"  Goal met: {'✅ Yes' if summary['goal_met'] else '❌ No'}")

    # Export
    record = autopilot.flight_record
    if args.export == "json":
        output = _format_json(record)
    elif args.export == "html":
        output = _generate_html(record)
    else:
        output = _format_text(record)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output)
        if not args.quiet:
            print(f"\n  📄 Report written to {args.output}")
    elif args.export != "text" or args.quiet:
        print(output)


if __name__ == "__main__":
    main()
