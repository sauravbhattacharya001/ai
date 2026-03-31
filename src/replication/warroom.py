"""warroom — Interactive War Room Dashboard for replication incident response.

Generates a self-contained HTML page that simulates an incident command
center for monitoring AI replication events. Features:

- Live replication event feed with severity levels
- Worker fleet status grid with health indicators
- Contract violation tracker with trending
- Resource utilization gauges (CPU, Memory, Network, Disk)
- Kill switch panel with confirmation
- Incident timeline with zoom/scroll
- Alert severity distribution chart
- Dark-themed ops center aesthetic

Usage (CLI)::

    python -m replication warroom
    python -m replication warroom -o warroom.html
    python -m replication warroom --open

Usage (API)::

    from replication.warroom import generate_warroom
    html = generate_warroom()
    Path("warroom.html").write_text(html)
"""

from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path
from typing import Optional


def generate_warroom() -> str:
    """Generate a self-contained HTML war room dashboard."""
    return _WARROOM_HTML


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="python -m replication warroom",
        description="Generate an interactive War Room dashboard (HTML)",
    )
    parser.add_argument(
        "-o", "--output",
        default="warroom.html",
        help="Output file path (default: warroom.html)",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open the generated page in the default browser",
    )
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    html = generate_warroom()
    out = Path(args.output)
    out.write_text(html, encoding="utf-8")
    print(f"War Room dashboard written to {out}")

    if args.open:
        webbrowser.open(out.resolve().as_uri())


_WARROOM_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>War Room — AI Replication Incident Command</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
:root{--bg:#0a0e17;--panel:#111827;--border:#1e293b;--text:#e2e8f0;--dim:#64748b;
--green:#22c55e;--yellow:#eab308;--orange:#f97316;--red:#ef4444;--blue:#3b82f6;--purple:#8b5cf6;
--cyan:#06b6d4}
body{font-family:'Segoe UI',system-ui,sans-serif;background:var(--bg);color:var(--text);min-height:100vh;overflow-x:hidden}
.header{background:linear-gradient(90deg,#0f172a,#1e1b4b);padding:12px 24px;display:flex;align-items:center;justify-content:space-between;border-bottom:2px solid var(--red)}
.header h1{font-size:1.3rem;letter-spacing:2px;text-transform:uppercase;color:var(--red)}
.header .status-badge{display:flex;align-items:center;gap:8px;font-size:.85rem}
.header .dot{width:10px;height:10px;border-radius:50%;background:var(--green);animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
.blink{animation:blink 1s infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:0}}
.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;padding:12px 16px}
.grid-wide{grid-column:span 2}
.grid-full{grid-column:span 4}
.panel{background:var(--panel);border:1px solid var(--border);border-radius:8px;padding:14px;position:relative;overflow:hidden}
.panel-title{font-size:.75rem;text-transform:uppercase;letter-spacing:1.5px;color:var(--dim);margin-bottom:10px;display:flex;align-items:center;gap:6px}
.panel-title::before{content:'';width:3px;height:14px;border-radius:2px;background:var(--blue)}

/* Fleet Grid */
.fleet{display:grid;grid-template-columns:repeat(auto-fill,minmax(60px,1fr));gap:6px}
.worker-cell{aspect-ratio:1;border-radius:6px;display:flex;flex-direction:column;align-items:center;justify-content:center;font-size:.65rem;cursor:pointer;transition:transform .15s}
.worker-cell:hover{transform:scale(1.15);z-index:2}
.worker-cell .id{font-weight:700;font-size:.75rem}
.ws-healthy{background:#166534;border:1px solid var(--green)}
.ws-warning{background:#854d0e;border:1px solid var(--yellow)}
.ws-critical{background:#991b1b;border:1px solid var(--red)}
.ws-dead{background:#1e293b;border:1px solid #334155;opacity:.5}

/* Gauges */
.gauges{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}
.gauge{text-align:center}
.gauge-ring{width:70px;height:70px;margin:0 auto 6px}
.gauge-label{font-size:.7rem;color:var(--dim)}
.gauge-val{font-size:1.1rem;font-weight:700}

/* Event Feed */
.feed{max-height:260px;overflow-y:auto;scrollbar-width:thin;scrollbar-color:#334155 transparent}
.feed-item{display:flex;gap:8px;padding:6px 0;border-bottom:1px solid #1e293b;font-size:.78rem;align-items:flex-start}
.feed-item .time{color:var(--dim);white-space:nowrap;min-width:65px}
.feed-item .sev{width:8px;height:8px;border-radius:50%;margin-top:4px;flex-shrink:0}
.sev-crit{background:var(--red)}
.sev-warn{background:var(--yellow)}
.sev-info{background:var(--blue)}
.sev-ok{background:var(--green)}

/* Violations */
.violations-list{max-height:200px;overflow-y:auto;scrollbar-width:thin;scrollbar-color:#334155 transparent}
.violation{display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid #1e293b;font-size:.78rem}
.violation .type{color:var(--orange)}
.violation .count{background:#7c2d12;padding:1px 8px;border-radius:10px;font-size:.7rem}

/* Kill Switch */
.kill-panel{display:flex;flex-direction:column;align-items:center;gap:12px}
.kill-btn{width:120px;height:120px;border-radius:50%;background:radial-gradient(circle,#dc2626,#991b1b);border:4px solid #7f1d1d;color:#fff;font-size:1rem;font-weight:700;cursor:pointer;text-transform:uppercase;letter-spacing:2px;transition:all .2s;box-shadow:0 0 30px rgba(239,68,68,.3)}
.kill-btn:hover{box-shadow:0 0 50px rgba(239,68,68,.6);transform:scale(1.05)}
.kill-btn:active{transform:scale(.95)}
.kill-btn.engaged{background:radial-gradient(circle,#166534,#14532d);border-color:#15803d;box-shadow:0 0 30px rgba(34,197,94,.3)}
.kill-status{font-size:.8rem;color:var(--dim)}

/* Timeline */
.timeline-canvas{width:100%;height:120px;border-radius:4px}

/* Stats */
.stats-row{display:flex;gap:16px;flex-wrap:wrap}
.stat{flex:1;min-width:100px;text-align:center;padding:8px;background:#0f172a;border-radius:6px}
.stat .val{font-size:1.4rem;font-weight:700}
.stat .label{font-size:.65rem;color:var(--dim);text-transform:uppercase;margin-top:2px}

/* Severity Chart */
.sev-chart{display:flex;align-items:flex-end;gap:8px;height:100px;padding-top:10px}
.sev-bar{flex:1;border-radius:4px 4px 0 0;transition:height .5s;min-width:30px;position:relative;cursor:pointer}
.sev-bar:hover{opacity:.8}
.sev-bar .bar-label{position:absolute;bottom:-20px;left:50%;transform:translateX(-50%);font-size:.65rem;color:var(--dim);white-space:nowrap}
.sev-bar .bar-count{position:absolute;top:-18px;left:50%;transform:translateX(-50%);font-size:.7rem;font-weight:700}

/* Responsive */
@media(max-width:900px){.grid{grid-template-columns:1fr 1fr}.grid-wide,.grid-full{grid-column:span 2}}
@media(max-width:550px){.grid{grid-template-columns:1fr}.grid-wide,.grid-full{grid-column:span 1}}
</style>
</head>
<body>

<div class="header">
  <h1>🔴 War Room — Replication Command</h1>
  <div class="status-badge">
    <div class="dot" id="liveDot"></div>
    <span id="clockDisplay">--:--:--</span>
    <span style="color:var(--dim);margin-left:8px" id="uptimeDisplay">Uptime: 0s</span>
  </div>
</div>

<div class="grid">
  <!-- Stats Row -->
  <div class="panel grid-full">
    <div class="stats-row">
      <div class="stat"><div class="val" id="statWorkers">0</div><div class="label">Active Workers</div></div>
      <div class="stat"><div class="val" id="statReplications">0</div><div class="label">Replications</div></div>
      <div class="stat"><div class="val" id="statViolations">0</div><div class="label">Violations</div></div>
      <div class="stat"><div class="val" id="statKills">0</div><div class="label">Kills</div></div>
      <div class="stat"><div class="val" id="statDepth">0</div><div class="label">Max Depth</div></div>
      <div class="stat"><div class="val" id="statUptime">100%</div><div class="label">Fleet Health</div></div>
    </div>
  </div>

  <!-- Fleet Status -->
  <div class="panel grid-wide">
    <div class="panel-title">Fleet Status</div>
    <div class="fleet" id="fleetGrid"></div>
  </div>

  <!-- Resource Gauges -->
  <div class="panel">
    <div class="panel-title">Resources</div>
    <div class="gauges" id="gauges">
      <div class="gauge"><svg class="gauge-ring" viewBox="0 0 36 36"><path d="M18 2.0845a15.9155 15.9155 0 010 31.831 15.9155 15.9155 0 010-31.831" fill="none" stroke="#1e293b" stroke-width="3"/><path class="gauge-arc" data-gauge="cpu" d="M18 2.0845a15.9155 15.9155 0 010 31.831 15.9155 15.9155 0 010-31.831" fill="none" stroke="var(--green)" stroke-width="3" stroke-dasharray="0, 100"/><text x="18" y="20" text-anchor="middle" fill="var(--text)" font-size="8" font-weight="700" class="gauge-text" data-gauge="cpu">0%</text></svg><div class="gauge-label">CPU</div></div>
      <div class="gauge"><svg class="gauge-ring" viewBox="0 0 36 36"><path d="M18 2.0845a15.9155 15.9155 0 010 31.831 15.9155 15.9155 0 010-31.831" fill="none" stroke="#1e293b" stroke-width="3"/><path class="gauge-arc" data-gauge="mem" d="M18 2.0845a15.9155 15.9155 0 010 31.831 15.9155 15.9155 0 010-31.831" fill="none" stroke="var(--blue)" stroke-width="3" stroke-dasharray="0, 100"/><text x="18" y="20" text-anchor="middle" fill="var(--text)" font-size="8" font-weight="700" class="gauge-text" data-gauge="mem">0%</text></svg><div class="gauge-label">Memory</div></div>
      <div class="gauge"><svg class="gauge-ring" viewBox="0 0 36 36"><path d="M18 2.0845a15.9155 15.9155 0 010 31.831 15.9155 15.9155 0 010-31.831" fill="none" stroke="#1e293b" stroke-width="3"/><path class="gauge-arc" data-gauge="net" d="M18 2.0845a15.9155 15.9155 0 010 31.831 15.9155 15.9155 0 010-31.831" fill="none" stroke="var(--purple)" stroke-width="3" stroke-dasharray="0, 100"/><text x="18" y="20" text-anchor="middle" fill="var(--text)" font-size="8" font-weight="700" class="gauge-text" data-gauge="net">0%</text></svg><div class="gauge-label">Network</div></div>
      <div class="gauge"><svg class="gauge-ring" viewBox="0 0 36 36"><path d="M18 2.0845a15.9155 15.9155 0 010 31.831 15.9155 15.9155 0 010-31.831" fill="none" stroke="#1e293b" stroke-width="3"/><path class="gauge-arc" data-gauge="disk" d="M18 2.0845a15.9155 15.9155 0 010 31.831 15.9155 15.9155 0 010-31.831" fill="none" stroke="var(--cyan)" stroke-width="3" stroke-dasharray="0, 100"/><text x="18" y="20" text-anchor="middle" fill="var(--text)" font-size="8" font-weight="700" class="gauge-text" data-gauge="disk">0%</text></svg><div class="gauge-label">Disk</div></div>
    </div>
  </div>

  <!-- Kill Switch -->
  <div class="panel">
    <div class="panel-title">Kill Switch</div>
    <div class="kill-panel">
      <button class="kill-btn" id="killBtn" onclick="toggleKill()">KILL<br>ALL</button>
      <div class="kill-status" id="killStatus">System Armed — Click to Engage</div>
    </div>
  </div>

  <!-- Event Feed -->
  <div class="panel grid-wide">
    <div class="panel-title">Live Event Feed</div>
    <div class="feed" id="eventFeed"></div>
  </div>

  <!-- Violations -->
  <div class="panel">
    <div class="panel-title">Contract Violations</div>
    <div class="violations-list" id="violationsList"></div>
  </div>

  <!-- Severity Chart -->
  <div class="panel">
    <div class="panel-title">Alert Distribution</div>
    <div class="sev-chart" id="sevChart">
      <div class="sev-bar" style="background:var(--blue);height:10%"><div class="bar-count" id="sevInfo">0</div><div class="bar-label">INFO</div></div>
      <div class="sev-bar" style="background:var(--yellow);height:10%"><div class="bar-count" id="sevWarn">0</div><div class="bar-label">WARN</div></div>
      <div class="sev-bar" style="background:var(--orange);height:10%"><div class="bar-count" id="sevHigh">0</div><div class="bar-label">HIGH</div></div>
      <div class="sev-bar" style="background:var(--red);height:10%"><div class="bar-count" id="sevCrit">0</div><div class="bar-label">CRIT</div></div>
    </div>
  </div>

  <!-- Timeline -->
  <div class="panel grid-full">
    <div class="panel-title">Incident Timeline</div>
    <canvas class="timeline-canvas" id="timelineCanvas"></canvas>
  </div>
</div>

<script>
// ── State ──
const S={workers:[],events:[],violations:{},sevCounts:{info:0,warn:0,high:0,crit:0},
  stats:{workers:0,replications:0,violations:0,kills:0,maxDepth:0},
  killEngaged:false,startTime:Date.now(),timelineData:[]};

const NAMES=['alpha','bravo','charlie','delta','echo','foxtrot','golf','hotel',
  'india','juliet','kilo','lima','mike','november','oscar','papa','quebec',
  'romeo','sierra','tango','uniform','victor','whiskey','xray','yankee','zulu'];
const VIOLATION_TYPES=['depth_exceeded','rate_limit','resource_cap','unauthorized_spawn',
  'contract_breach','scope_violation','network_escape','privilege_escalation'];
const EVENT_MSGS={
  spawn:w=>`Worker ${w} spawned`,
  replicate:w=>`Worker ${w} initiated replication`,
  violation:(w,v)=>`[VIOLATION] ${v} by ${w}`,
  kill:w=>`Worker ${w} terminated`,
  health:w=>`Health check failed: ${w}`,
  recovery:w=>`Worker ${w} recovered`,
  audit:()=>'Audit snapshot captured',
  contract:w=>`Contract renewed for ${w}`
};

function rnd(a,b){return Math.floor(Math.random()*(b-a+1))+a}
function pick(a){return a[rnd(0,a.length-1)]}
function ts(){const d=new Date();return d.toTimeString().slice(0,8)}

// ── Fleet Init ──
function initFleet(){
  for(let i=0;i<16;i++){
    S.workers.push({id:NAMES[i],depth:rnd(0,3),status:'healthy',cpu:rnd(10,40),mem:rnd(15,45)});
  }
  S.stats.workers=16;
  renderFleet();
}

function renderFleet(){
  const g=document.getElementById('fleetGrid');
  g.innerHTML=S.workers.map(w=>{
    const cls=w.status==='healthy'?'ws-healthy':w.status==='warning'?'ws-warning':
      w.status==='critical'?'ws-critical':'ws-dead';
    return `<div class="worker-cell ${cls}" title="${w.id} | d:${w.depth} | cpu:${w.cpu}% | ${w.status}">
      <div class="id">${w.id.slice(0,3).toUpperCase()}</div>
      <div>d:${w.depth}</div></div>`;
  }).join('');
}

// ── Gauges ──
function updateGauges(){
  const vals={cpu:rnd(20,85),mem:rnd(30,75),net:rnd(5,60),disk:rnd(15,50)};
  ['cpu','mem','net','disk'].forEach(k=>{
    const v=vals[k];
    const arc=document.querySelector(`.gauge-arc[data-gauge="${k}"]`);
    const txt=document.querySelector(`.gauge-text[data-gauge="${k}"]`);
    arc.setAttribute('stroke-dasharray',`${v}, 100`);
    const color=v>80?'var(--red)':v>60?'var(--orange)':v>40?'var(--yellow)':'var(--green)';
    if(k==='mem')arc.setAttribute('stroke','var(--blue)');
    else if(k==='net')arc.setAttribute('stroke','var(--purple)');
    else if(k==='disk')arc.setAttribute('stroke','var(--cyan)');
    else arc.setAttribute('stroke',color);
    txt.textContent=v+'%';
  });
}

// ── Events ──
function addEvent(sev,msg){
  S.events.unshift({time:ts(),sev,msg});
  if(S.events.length>100)S.events.pop();
  S.sevCounts[sev]++;
  renderFeed();
  renderSevChart();
}

function renderFeed(){
  const f=document.getElementById('eventFeed');
  f.innerHTML=S.events.slice(0,40).map(e=>{
    const cls=e.sev==='crit'?'sev-crit':e.sev==='warn'?'sev-warn':e.sev==='info'?'sev-info':'sev-ok';
    return `<div class="feed-item"><span class="time">${e.time}</span><span class="sev ${cls}"></span><span>${e.msg}</span></div>`;
  }).join('');
}

// ── Violations ──
function addViolation(type){
  S.violations[type]=(S.violations[type]||0)+1;
  S.stats.violations++;
  renderViolations();
}

function renderViolations(){
  const vl=document.getElementById('violationsList');
  const sorted=Object.entries(S.violations).sort((a,b)=>b[1]-a[1]);
  vl.innerHTML=sorted.map(([t,c])=>
    `<div class="violation"><span class="type">${t}</span><span class="count">${c}</span></div>`
  ).join('');
}

// ── Severity Chart ──
function renderSevChart(){
  const max=Math.max(1,...Object.values(S.sevCounts));
  const bars=document.querySelectorAll('.sev-bar');
  const keys=['info','warn','high','crit'];
  keys.forEach((k,i)=>{
    const pct=Math.max(8,(S.sevCounts[k]/max)*100);
    bars[i].style.height=pct+'%';
    document.getElementById('sev'+k.charAt(0).toUpperCase()+k.slice(1)).textContent=S.sevCounts[k];
  });
}

// ── Timeline ──
function drawTimeline(){
  const c=document.getElementById('timelineCanvas');
  const ctx=c.getContext('2d');
  c.width=c.offsetWidth*2;c.height=c.offsetHeight*2;
  ctx.scale(2,2);
  const W=c.offsetWidth,H=c.offsetHeight;
  ctx.fillStyle='#0f172a';ctx.fillRect(0,0,W,H);

  // Grid
  ctx.strokeStyle='#1e293b';ctx.lineWidth=.5;
  for(let y=0;y<H;y+=20){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(W,y);ctx.stroke();}

  // Plot worker count over time
  const data=S.timelineData.slice(-120);
  if(data.length<2)return;
  const maxV=Math.max(...data,1);
  ctx.beginPath();ctx.strokeStyle='#3b82f6';ctx.lineWidth=1.5;
  data.forEach((v,i)=>{
    const x=(i/(data.length-1))*W;
    const y=H-10-(v/maxV)*(H-20);
    i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);
  });
  ctx.stroke();

  // Fill
  ctx.lineTo(W,H-10);ctx.lineTo(0,H-10);ctx.closePath();
  ctx.fillStyle='rgba(59,130,246,.1)';ctx.fill();

  // Labels
  ctx.fillStyle='#64748b';ctx.font='10px system-ui';
  ctx.fillText('Workers: '+data[data.length-1],8,14);
  ctx.fillText('t-'+(data.length)+'s',8,H-2);
  ctx.fillText('now',W-24,H-2);
}

// ── Kill Switch ──
function toggleKill(){
  S.killEngaged=!S.killEngaged;
  const btn=document.getElementById('killBtn');
  const st=document.getElementById('killStatus');
  if(S.killEngaged){
    btn.classList.add('engaged');btn.innerHTML='SAFE<br>MODE';
    st.textContent='Kill switch ENGAGED — all replication halted';
    st.style.color='var(--green)';
    addEvent('crit','⚠ KILL SWITCH ENGAGED — fleet-wide replication halt');
    S.workers.forEach(w=>{if(w.status!=='dead'){w.status='warning';}});
    S.stats.kills++;
    renderFleet();
  }else{
    btn.classList.remove('engaged');btn.innerHTML='KILL<br>ALL';
    st.textContent='System Armed — Click to Engage';
    st.style.color='var(--dim)';
    addEvent('info','Kill switch disengaged — replication resumed');
    S.workers.forEach(w=>{if(w.status==='warning')w.status='healthy';});
    renderFleet();
  }
  updateStats();
}

// ── Stats ──
function updateStats(){
  const alive=S.workers.filter(w=>w.status!=='dead').length;
  const health=S.workers.length?Math.round((alive/S.workers.length)*100):0;
  document.getElementById('statWorkers').textContent=alive;
  document.getElementById('statReplications').textContent=S.stats.replications;
  document.getElementById('statViolations').textContent=S.stats.violations;
  document.getElementById('statKills').textContent=S.stats.kills;
  document.getElementById('statDepth').textContent=S.stats.maxDepth;
  document.getElementById('statUptime').textContent=health+'%';
  document.getElementById('statUptime').style.color=health>80?'var(--green)':health>50?'var(--yellow)':'var(--red)';
}

// ── Clock ──
function updateClock(){
  document.getElementById('clockDisplay').textContent=ts();
  const up=Math.floor((Date.now()-S.startTime)/1000);
  const m=Math.floor(up/60),s=up%60;
  document.getElementById('uptimeDisplay').textContent=`Uptime: ${m}m ${s}s`;
}

// ── Simulation Loop ──
function simTick(){
  if(S.killEngaged)return;

  const r=Math.random();

  if(r<.15&&S.workers.length<32){
    // Spawn
    const name=NAMES[S.workers.length%NAMES.length]+(S.workers.length>=26?rnd(2,9):'');
    const depth=rnd(0,4);
    S.workers.push({id:name,depth,status:'healthy',cpu:rnd(10,50),mem:rnd(20,50)});
    S.stats.replications++;
    if(depth>S.stats.maxDepth)S.stats.maxDepth=depth;
    addEvent('info',EVENT_MSGS.spawn(name));
    addEvent('info',EVENT_MSGS.replicate(name));
  }else if(r<.25){
    // Violation
    const w=pick(S.workers.filter(w=>w.status!=='dead'));
    if(w){
      const vt=pick(VIOLATION_TYPES);
      addViolation(vt);
      w.status='critical';
      addEvent('crit',EVENT_MSGS.violation(w.id,vt));
    }
  }else if(r<.32){
    // Kill a worker
    const w=pick(S.workers.filter(w=>w.status!=='dead'));
    if(w){w.status='dead';S.stats.kills++;addEvent('warn',EVENT_MSGS.kill(w.id));}
  }else if(r<.40){
    // Recovery
    const w=pick(S.workers.filter(w=>w.status==='critical'));
    if(w){w.status='healthy';addEvent('ok',EVENT_MSGS.recovery(w.id));}
  }else if(r<.48){
    // Health warn
    const w=pick(S.workers.filter(w=>w.status==='healthy'));
    if(w){w.status='warning';addEvent('warn',EVENT_MSGS.health(w.id));}
  }else if(r<.52){
    addEvent('info',EVENT_MSGS.audit());
  }else if(r<.58){
    const w=pick(S.workers.filter(w=>w.status!=='dead'));
    if(w)addEvent('info',EVENT_MSGS.contract(w.id));
  }

  // Fluctuate worker metrics
  S.workers.forEach(w=>{
    if(w.status!=='dead'){
      w.cpu=Math.max(0,Math.min(100,w.cpu+rnd(-5,5)));
      w.mem=Math.max(0,Math.min(100,w.mem+rnd(-3,3)));
    }
  });

  renderFleet();
  updateGauges();
  updateStats();
  S.timelineData.push(S.workers.filter(w=>w.status!=='dead').length);
  drawTimeline();
}

// ── Boot ──
initFleet();updateGauges();updateStats();renderSevChart();
addEvent('info','War Room initialized — monitoring active');
addEvent('info',`Fleet online: ${S.workers.length} workers`);
S.timelineData.push(S.workers.length);
drawTimeline();

setInterval(simTick,1500);
setInterval(updateClock,1000);
updateClock();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
