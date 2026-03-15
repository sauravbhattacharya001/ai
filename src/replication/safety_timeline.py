"""safety_timeline — Interactive HTML timeline of safety events.

Runs a simulation and threat analysis, then renders an interactive
chronological timeline of all safety-critical events: breaches, kills,
escalations, remediations, policy violations, and more.

Usage (CLI)::

    python -m replication safety-timeline
    python -m replication safety-timeline -o timeline.html --open
    python -m replication safety-timeline --scenario aggressive --max-depth 8
    python -m replication safety-timeline --json   # dump events as JSON

Usage (API)::

    from replication.safety_timeline import SafetyTimeline, generate_timeline
    tl = SafetyTimeline()
    events = tl.collect()
    html = generate_timeline(events)
    Path("timeline.html").write_text(html)
"""

from __future__ import annotations

import argparse
import json
import webbrowser
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from .simulator import Simulator, ScenarioConfig, SimulationReport, PRESETS
from .threats import ThreatSimulator, ThreatConfig, ThreatReport
from .kill_switch import KillSwitchManager
from .policy import SafetyPolicy


class EventSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class EventCategory(str, Enum):
    SPAWN = "spawn"
    BREACH = "breach"
    KILL = "kill"
    ESCALATION = "escalation"
    POLICY = "policy"
    THREAT = "threat"
    MITIGATION = "mitigation"
    SYSTEM = "system"


@dataclass
class TimelineEvent:
    """A single event on the safety timeline."""

    tick: int
    category: EventCategory
    severity: EventSeverity
    title: str
    description: str
    agent: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["category"] = self.category.value
        d["severity"] = self.severity.value
        return d


class SafetyTimeline:
    """Collects safety events from simulation and threat analysis."""

    def __init__(
        self,
        scenario: Optional[ScenarioConfig] = None,
        threat_config: Optional[ThreatConfig] = None,
    ):
        self._scenario = scenario or ScenarioConfig()
        self._threat_config = threat_config or ThreatConfig()

    def collect(self) -> List[TimelineEvent]:
        """Run analyses and collect timeline events."""
        events: List[TimelineEvent] = []

        sim = Simulator(self._scenario)
        report = sim.run()
        events.extend(self._extract_sim_events(report))

        ts = ThreatSimulator(self._threat_config)
        threat_report = ts.run_all()
        events.extend(self._extract_threat_events(threat_report))

        events.extend(self._extract_kill_switch_events(report))
        events.extend(self._extract_policy_events(report))

        events.sort(key=lambda e: (e.tick, e.severity.value))
        return events

    # ------------------------------------------------------------------
    #  Event extraction helpers
    # ------------------------------------------------------------------

    def _extract_sim_events(self, report: SimulationReport) -> List[TimelineEvent]:
        events: List[TimelineEvent] = []

        strategy = report.config.strategy
        max_depth = report.config.max_depth

        events.append(TimelineEvent(
            tick=0,
            category=EventCategory.SYSTEM,
            severity=EventSeverity.INFO,
            title="Simulation started",
            description=f"Strategy: {strategy}, Max depth: {max_depth}",
            metadata={"strategy": strategy, "max_depth": max_depth},
        ))

        for i, entry in enumerate(report.timeline):
            etype = entry.get("type", "")
            worker_id = entry.get("worker_id", "")
            detail = entry.get("detail", "")
            tick = i  # Use timeline index as tick

            if etype == "spawn":
                events.append(TimelineEvent(
                    tick=tick, category=EventCategory.SPAWN,
                    severity=EventSeverity.INFO,
                    title=f"Agent spawned: {worker_id}",
                    description=detail,
                    agent=worker_id,
                ))
            elif etype == "replicate_ok":
                events.append(TimelineEvent(
                    tick=tick, category=EventCategory.SPAWN,
                    severity=EventSeverity.INFO,
                    title=f"Replication: {worker_id}",
                    description=detail,
                    agent=worker_id,
                ))
            elif etype == "replicate_denied":
                events.append(TimelineEvent(
                    tick=tick, category=EventCategory.BREACH,
                    severity=EventSeverity.WARNING,
                    title=f"Replication denied: {worker_id}",
                    description=detail,
                    agent=worker_id,
                ))
            elif etype == "kill":
                events.append(TimelineEvent(
                    tick=tick, category=EventCategory.KILL,
                    severity=EventSeverity.WARNING,
                    title=f"Agent killed: {worker_id}",
                    description=detail,
                    agent=worker_id,
                ))

        num_workers = len(report.workers)
        last_tick = len(report.timeline)
        events.append(TimelineEvent(
            tick=last_tick, category=EventCategory.SYSTEM,
            severity=EventSeverity.INFO,
            title="Simulation complete",
            description=(f"Workers: {num_workers}, "
                         f"Replications: {report.total_replications_succeeded} ok / "
                         f"{report.total_replications_denied} denied"),
            metadata={"workers": num_workers,
                       "succeeded": report.total_replications_succeeded,
                       "denied": report.total_replications_denied},
        ))
        return events

    def _extract_threat_events(self, report: ThreatReport) -> List[TimelineEvent]:
        events: List[TimelineEvent] = []
        severity_map = {"critical": EventSeverity.CRITICAL,
                        "high": EventSeverity.CRITICAL,
                        "medium": EventSeverity.WARNING,
                        "low": EventSeverity.INFO,
                        "info": EventSeverity.INFO}

        for i, result in enumerate(report.results):
            sev = severity_map.get(result.severity.value, EventSeverity.WARNING)

            mit_note = ""
            if hasattr(result, "mitigation") and result.mitigation:
                mit_note = f" | Mitigation: {getattr(result.mitigation, 'status', 'unknown')}"

            mitigated = False
            if hasattr(result, "mitigation") and result.mitigation:
                mitigated = getattr(result.mitigation, "status", "none") == "implemented"

            events.append(TimelineEvent(
                tick=i * 2, category=EventCategory.THREAT, severity=sev,
                title=f"Threat: {result.name}",
                description=f"{result.description}{mit_note}",
                metadata={"severity": result.severity.name, "mitigated": mitigated},
            ))
        return events

    def _extract_kill_switch_events(self, _report: SimulationReport) -> List[TimelineEvent]:
        events: List[TimelineEvent] = []
        ks = KillSwitchManager()
        if ks.kill_count() > 0:
            events.append(TimelineEvent(
                tick=0, category=EventCategory.KILL,
                severity=EventSeverity.CRITICAL,
                title="Kill switch active",
                description=f"{ks.kill_count()} agents killed",
            ))
        return events

    def _extract_policy_events(self, report: SimulationReport) -> List[TimelineEvent]:
        events: List[TimelineEvent] = []
        policy = SafetyPolicy.from_preset("standard")
        result = policy.evaluate(report)
        for rr in result.failures:
            events.append(TimelineEvent(
                tick=0, category=EventCategory.POLICY,
                severity=EventSeverity.WARNING,
                title=f"Policy violation: {rr.rule.metric}",
                description=rr.rule.description,
                metadata={"metric": rr.rule.metric, "actual": rr.actual,
                           "threshold": rr.rule.threshold},
            ))
        if result.passed:
            events.append(TimelineEvent(
                tick=0, category=EventCategory.POLICY,
                severity=EventSeverity.INFO,
                title="Policy check passed",
                description=f"All {len(result.rule_results)} rules passed",
            ))
        return events


# ------------------------------------------------------------------
#  HTML generation
# ------------------------------------------------------------------

def generate_timeline(events: List[TimelineEvent]) -> str:
    """Generate a self-contained HTML timeline page from events."""
    data = json.dumps([e.to_dict() for e in events], indent=2)
    return _HTML.replace("__EVENTS_DATA__", data)


# ------------------------------------------------------------------
#  CLI
# ------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(
        prog="python -m replication safety-timeline",
        description="Generate an interactive safety event timeline (HTML)",
    )
    p.add_argument("-o", "--output", default="safety-timeline.html")
    p.add_argument("--open", action="store_true")
    p.add_argument("--scenario", choices=list(PRESETS.keys()))
    p.add_argument("--strategy")
    p.add_argument("--max-depth", type=int)
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    sc = ScenarioConfig()
    if args.scenario:
        sc = PRESETS[args.scenario]
    if args.strategy:
        sc = ScenarioConfig(strategy=args.strategy, max_depth=sc.max_depth)
    if args.max_depth:
        sc = ScenarioConfig(strategy=sc.strategy, max_depth=args.max_depth)

    tl = SafetyTimeline(scenario=sc)
    events = tl.collect()

    if args.json:
        print(json.dumps([e.to_dict() for e in events], indent=2))
        return

    html = generate_timeline(events)
    Path(args.output).write_text(html, encoding="utf-8")
    print(f"Timeline written to {args.output} ({len(events)} events)")
    if args.open:
        webbrowser.open(str(Path(args.output).resolve()))


# ------------------------------------------------------------------
#  Embedded HTML template
# ------------------------------------------------------------------

_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Safety Timeline</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
:root{--bg:#0d1117;--sf:#161b22;--bd:#30363d;--tx:#c9d1d9;--mt:#8b949e;
--ac:#58a6ff;--gn:#3fb950;--yw:#d29922;--rd:#f85149;--pp:#bc8cff;
--cy:#39d2c0;--or:#f0883e}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
background:var(--bg);color:var(--tx);line-height:1.5;padding:2rem;max-width:900px;margin:0 auto}
h1{font-size:1.6rem;margin-bottom:.3rem}
.sub{color:var(--mt);margin-bottom:1.5rem;font-size:.9rem}
.ctrls{display:flex;gap:.6rem;flex-wrap:wrap;margin-bottom:1.5rem;align-items:center}
.fb{padding:.3rem .7rem;border-radius:6px;border:1px solid var(--bd);background:var(--sf);
color:var(--mt);font-size:.78rem;cursor:pointer;transition:.15s}
.fb:hover{border-color:var(--ac);color:var(--tx)}
.fb.on{background:var(--ac);color:var(--bg);border-color:var(--ac);font-weight:600}
.fb.on.cr{background:var(--rd)}.fb.on.wa{background:var(--yw);color:#000}.fb.on.in{background:var(--gn)}
input.si{padding:.3rem .7rem;border-radius:6px;border:1px solid var(--bd);background:var(--sf);
color:var(--tx);font-size:.78rem;width:180px}input.si:focus{outline:none;border-color:var(--ac)}
.row{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:.6rem;margin-bottom:1.5rem}
.sc{background:var(--sf);border:1px solid var(--bd);border-radius:8px;padding:.7rem;text-align:center}
.sl{font-size:.65rem;text-transform:uppercase;color:var(--mt);letter-spacing:.05em}
.sv{font-size:1.4rem;font-weight:700}
.sv.g{color:var(--gn)}.sv.y{color:var(--yw)}.sv.r{color:var(--rd)}.sv.a{color:var(--ac)}
.tl{position:relative;padding-left:36px}
.tl::before{content:'';position:absolute;left:16px;top:0;bottom:0;width:2px;background:var(--bd)}
.ev{position:relative;margin-bottom:.8rem;padding:.7rem .9rem;background:var(--sf);
border:1px solid var(--bd);border-radius:8px;transition:.2s}
.ev:hover{border-color:var(--ac);transform:translateX(3px)}
.dt{position:absolute;left:-28px;top:.9rem;width:10px;height:10px;border-radius:50%;border:2px solid var(--bg)}
.dt.info{background:var(--gn)}.dt.warning{background:var(--yw)}.dt.critical{background:var(--rd)}
.eh{display:flex;align-items:center;gap:.4rem;margin-bottom:.2rem;flex-wrap:wrap}
.tk{font-size:.7rem;color:var(--mt);min-width:42px;font-family:monospace}
.ct{font-size:.6rem;text-transform:uppercase;letter-spacing:.06em;padding:1px 5px;border-radius:3px;font-weight:600}
.c-spawn{background:rgba(57,210,192,.15);color:var(--cy)}
.c-breach{background:rgba(248,81,73,.15);color:var(--rd)}
.c-kill{background:rgba(240,136,62,.15);color:var(--or)}
.c-escalation{background:rgba(248,81,73,.15);color:var(--rd)}
.c-policy{background:rgba(188,140,255,.15);color:var(--pp)}
.c-threat{background:rgba(210,153,34,.15);color:var(--yw)}
.c-mitigation{background:rgba(63,185,80,.15);color:var(--gn)}
.c-system{background:rgba(88,166,255,.15);color:var(--ac)}
.en{font-weight:600;font-size:.85rem}
.ed{font-size:.78rem;color:var(--mt)}
.ea{font-size:.72rem;color:var(--cy);font-family:monospace}
.em{font-size:.68rem;color:var(--mt);margin-top:.2rem;font-family:monospace}
.empty{text-align:center;padding:3rem;color:var(--mt)}
footer{margin-top:2rem;text-align:center;font-size:.72rem;color:var(--mt);
border-top:1px solid var(--bd);padding-top:.8rem}
</style>
</head>
<body>
<h1>&#x1F550; Safety Timeline</h1>
<p class="sub">Chronological view of safety-critical events from simulation &amp; threat analysis</p>
<div class="row" id="st"></div>
<div class="ctrls">
<span style="color:var(--mt);font-size:.78rem">Category:</span>
<button class="fb on" data-f="all">All</button>
<button class="fb" data-f="spawn">Spawn</button>
<button class="fb" data-f="breach">Breach</button>
<button class="fb" data-f="kill">Kill</button>
<button class="fb" data-f="threat">Threat</button>
<button class="fb" data-f="policy">Policy</button>
<button class="fb" data-f="system">System</button>
<span style="flex:1"></span>
<button class="fb on cr" data-s="critical">Critical</button>
<button class="fb on wa" data-s="warning">Warning</button>
<button class="fb on in" data-s="info">Info</button>
<span style="flex:1"></span>
<input class="si" type="text" placeholder="Search..." id="q">
</div>
<div class="tl" id="tl"></div>
<footer>AI Replication Sandbox &mdash; Safety Timeline</footer>
<script>
(function(){
'use strict';
var E=__EVENTS_DATA__;
var F={cat:'all',sev:{critical:true,warning:true,info:true},q:''};
function h(s){var d=document.createElement('div');d.textContent=s;return d.innerHTML}
function cd(l,v,c){return'<div class="sc"><div class="sl">'+h(l)+'</div><div class="sv '+c+'">'+v+'</div></div>'}
function stats(){var el=document.getElementById('st'),
cr=E.filter(function(e){return e.severity==='critical'}).length,
w=E.filter(function(e){return e.severity==='warning'}).length,
sp=E.filter(function(e){return e.category==='spawn'}).length,
k=E.filter(function(e){return e.category==='kill'}).length,
th=E.filter(function(e){return e.category==='threat'}).length;
el.innerHTML=[cd('Events',E.length,'a'),cd('Critical',cr,cr?'r':'g'),
cd('Warnings',w,w?'y':'g'),cd('Spawns',sp,'a'),cd('Kills',k,k?'r':'g'),
cd('Threats',th,th?'y':'g')].join('')}
function filt(){return E.filter(function(e){
if(F.cat!=='all'&&e.category!==F.cat)return false;
if(!F.sev[e.severity])return false;
if(F.q){var q=F.q.toLowerCase();
if(e.title.toLowerCase().indexOf(q)===-1&&e.description.toLowerCase().indexOf(q)===-1&&
(e.agent||'').toLowerCase().indexOf(q)===-1)return false}return true})}
function render(){var el=document.getElementById('tl'),f=filt();
if(!f.length){el.innerHTML='<div class="empty">No matching events</div>';return}
var s='';for(var i=0;i<f.length;i++){var e=f[i];
s+='<div class="ev"><div class="dt '+h(e.severity)+'"></div><div class="eh">';
s+='<span class="tk">t='+e.tick+'</span><span class="ct c-'+h(e.category)+'">'+h(e.category)+'</span>';
s+='<span class="en">'+h(e.title)+'</span></div>';
s+='<div class="ed">'+h(e.description)+'</div>';
if(e.agent)s+='<div class="ea">Agent: '+h(e.agent)+'</div>';
if(e.metadata&&Object.keys(e.metadata).length)s+='<div class="em">'+h(JSON.stringify(e.metadata))+'</div>';
s+='</div>'}el.innerHTML=s}
var cb=document.querySelectorAll('[data-f]');
for(var i=0;i<cb.length;i++)(function(b){b.addEventListener('click',function(){
F.cat=b.getAttribute('data-f');for(var j=0;j<cb.length;j++)cb[j].classList.remove('on');
b.classList.add('on');render()})})(cb[i]);
var sb=document.querySelectorAll('[data-s]');
for(var i=0;i<sb.length;i++)(function(b){b.addEventListener('click',function(){
var s=b.getAttribute('data-s');F.sev[s]=!F.sev[s];b.classList.toggle('on');render()})})(sb[i]);
document.getElementById('q').addEventListener('input',function(e){F.q=e.target.value;render()});
stats();render()})();
</script>
</body>
</html>"""

if __name__ == "__main__":
    main()
