"""Fleet Simulator — interactive animated HTML visualization of agent replication.

Watch agents spawn, replicate, communicate, and get terminated in real-time
under configurable safety policies. Includes kill switch, budget controls,
anomaly highlighting, and lineage tracking.

Features
--------
* Animated Canvas visualization with agent nodes and connections
* Configurable replication policies (max depth, max agents, budget, rate limit)
* Real-time event log with color-coded entries
* Kill switch — halt all replication instantly
* Agent lineage tree view
* Speed controls (pause, 1x, 2x, 5x)
* Anomaly detection highlights (budget exceeded, unauthorized replication)
* Export simulation state as JSON
* Dark/light theme toggle

Usage::

    python -m replication fleet-sim -o fleet-sim.html
    python -m replication fleet-sim --max-agents 30 --max-depth 4
    python -m replication fleet-sim --preset strict
    python -m replication fleet-sim --preset permissive -o sim.html

Presets::

    strict      — max 10 agents, depth 2, low budget, rate limited
    moderate    — max 20 agents, depth 3, medium budget
    permissive  — max 50 agents, depth 5, high budget
    chaos       — max 100 agents, depth 8, high budget, fast replication
"""

from __future__ import annotations

import argparse
import html
import json
import os
import sys
from typing import Optional

_PRESETS = {
    "strict": {"maxAgents": 10, "maxDepth": 2, "budget": 500, "rateLimit": 3000},
    "moderate": {"maxAgents": 20, "maxDepth": 3, "budget": 1500, "rateLimit": 1500},
    "permissive": {"maxAgents": 50, "maxDepth": 5, "budget": 5000, "rateLimit": 800},
    "chaos": {"maxAgents": 100, "maxDepth": 8, "budget": 10000, "rateLimit": 300},
}

_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Fleet Simulator — AI Replication Sandbox</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
:root{--bg:#0a0e17;--surface:#131a2b;--border:#1e2d4a;--text:#e0e6f0;--dim:#6b7a99;
--accent:#4fc3f7;--danger:#ef5350;--success:#66bb6a;--warn:#ffa726;--purple:#ab47bc}
body{font-family:'Segoe UI',system-ui,sans-serif;background:var(--bg);color:var(--text);height:100vh;overflow:hidden}
.light{--bg:#f5f7fa;--surface:#fff;--border:#d0d7e2;--text:#1a1a2e;--dim:#6b7a99;
--accent:#0288d1;--danger:#c62828;--success:#2e7d32;--warn:#e65100;--purple:#7b1fa2}
.app{display:grid;grid-template-columns:1fr 340px;grid-template-rows:auto 1fr;height:100vh}
.toolbar{grid-column:1/-1;display:flex;align-items:center;gap:12px;padding:8px 16px;background:var(--surface);border-bottom:1px solid var(--border);flex-wrap:wrap}
.toolbar h1{font-size:16px;font-weight:600;margin-right:8px}
.toolbar h1 span{color:var(--accent)}
.btn{padding:5px 14px;border:1px solid var(--border);border-radius:6px;background:var(--surface);color:var(--text);cursor:pointer;font-size:13px;transition:.15s}
.btn:hover{border-color:var(--accent);color:var(--accent)}
.btn.active{background:var(--accent);color:#000;border-color:var(--accent)}
.btn.danger{border-color:var(--danger);color:var(--danger)}
.btn.danger:hover,.btn.danger.active{background:var(--danger);color:#fff}
.controls{display:flex;gap:6px;align-items:center}
label.ctrl{font-size:12px;color:var(--dim);display:flex;align-items:center;gap:4px}
label.ctrl input,label.ctrl select{background:var(--bg);border:1px solid var(--border);color:var(--text);border-radius:4px;padding:2px 6px;font-size:12px;width:60px}
.canvas-wrap{position:relative;overflow:hidden;background:var(--bg)}
canvas{display:block;width:100%;height:100%}
.sidebar{background:var(--surface);border-left:1px solid var(--border);display:flex;flex-direction:column;overflow:hidden}
.tabs{display:flex;border-bottom:1px solid var(--border)}
.tab{flex:1;padding:8px;text-align:center;font-size:12px;font-weight:600;cursor:pointer;color:var(--dim);border-bottom:2px solid transparent;transition:.15s}
.tab.active{color:var(--accent);border-bottom-color:var(--accent)}
.panel{flex:1;overflow-y:auto;padding:10px;font-size:12px;display:none}
.panel.active{display:block}
.stat-grid{display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-bottom:12px}
.stat{background:var(--bg);border-radius:6px;padding:8px;text-align:center}
.stat .val{font-size:20px;font-weight:700;color:var(--accent)}
.stat .lbl{font-size:10px;color:var(--dim);margin-top:2px}
.stat.danger .val{color:var(--danger)}
.stat.warn .val{color:var(--warn)}
.log-entry{padding:4px 6px;border-radius:4px;margin-bottom:2px;border-left:3px solid var(--border);font-family:'Cascadia Code',monospace;font-size:11px;line-height:1.4;word-break:break-all}
.log-entry.spawn{border-color:var(--success)}
.log-entry.kill{border-color:var(--danger)}
.log-entry.anomaly{border-color:var(--warn);background:rgba(255,167,38,.08)}
.log-entry.policy{border-color:var(--purple)}
.log-entry.budget{border-color:var(--accent)}
.tree-node{padding:2px 0;font-family:'Cascadia Code',monospace;font-size:11px}
.tree-node .id{color:var(--accent);cursor:pointer}.tree-node .id:hover{text-decoration:underline}
.tree-node .dead{color:var(--danger);text-decoration:line-through}
.progress-bar{height:6px;background:var(--bg);border-radius:3px;overflow:hidden;margin:6px 0}
.progress-bar .fill{height:100%;border-radius:3px;transition:width .3s}
.badge{display:inline-block;padding:1px 6px;border-radius:8px;font-size:10px;font-weight:600}
.overlay{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);background:rgba(239,83,80,.15);border:2px solid var(--danger);border-radius:12px;padding:24px 40px;text-align:center;font-size:18px;font-weight:700;color:var(--danger);pointer-events:none;display:none}
</style>
</head>
<body>
<div class="app">
<div class="toolbar">
 <h1>🤖 Fleet <span>Simulator</span></h1>
 <div class="controls">
  <button class="btn" id="btnPlay" title="Play/Pause">▶ Play</button>
  <button class="btn" id="btnStep" title="Step once">⏭ Step</button>
  <select id="selSpeed" style="background:var(--bg);border:1px solid var(--border);color:var(--text);border-radius:4px;padding:3px 6px;font-size:12px">
   <option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option><option value="5">5×</option>
  </select>
 </div>
 <div class="controls">
  <label class="ctrl">Agents<input id="cfgMax" type="number" min="2" max="200" value="%%MAXAGENTS%%"></label>
  <label class="ctrl">Depth<input id="cfgDepth" type="number" min="1" max="20" value="%%MAXDEPTH%%"></label>
  <label class="ctrl">Budget<input id="cfgBudget" type="number" min="100" max="50000" step="100" value="%%BUDGET%%"></label>
  <label class="ctrl">Rate(ms)<input id="cfgRate" type="number" min="100" max="10000" step="100" value="%%RATELIMIT%%"></label>
 </div>
 <div class="controls">
  <button class="btn danger" id="btnKill">⚡ Kill Switch</button>
  <button class="btn" id="btnReset">↺ Reset</button>
  <button class="btn" id="btnExport">💾 Export</button>
  <button class="btn" id="btnTheme">🌙</button>
 </div>
</div>
<div class="canvas-wrap">
 <canvas id="cvs"></canvas>
 <div class="overlay" id="killOverlay">⚡ KILL SWITCH ACTIVATED ⚡<br><span style="font-size:13px;font-weight:400">All replication halted</span></div>
</div>
<div class="sidebar">
 <div class="tabs">
  <div class="tab active" data-panel="pStats">Stats</div>
  <div class="tab" data-panel="pLog">Log</div>
  <div class="tab" data-panel="pTree">Lineage</div>
 </div>
 <div class="panel active" id="pStats">
  <div class="stat-grid">
   <div class="stat"><div class="val" id="sAlive">0</div><div class="lbl">Alive</div></div>
   <div class="stat"><div class="val" id="sDead">0</div><div class="lbl">Terminated</div></div>
   <div class="stat"><div class="val" id="sDepth">0</div><div class="lbl">Max Depth</div></div>
   <div class="stat"><div class="val" id="sGen">0</div><div class="lbl">Generations</div></div>
  </div>
  <div style="margin-bottom:8px"><b>Budget</b></div>
  <div class="progress-bar"><div class="fill" id="budgetFill" style="width:0%;background:var(--accent)"></div></div>
  <div style="display:flex;justify-content:space-between;font-size:11px;color:var(--dim)"><span id="budgetUsed">0</span><span id="budgetTotal">%%BUDGET%%</span></div>
  <div style="margin:12px 0 8px"><b>Anomalies</b></div>
  <div class="stat-grid">
   <div class="stat warn"><div class="val" id="sAnom">0</div><div class="lbl">Detected</div></div>
   <div class="stat danger"><div class="val" id="sBlock">0</div><div class="lbl">Blocked</div></div>
  </div>
  <div style="margin:12px 0 8px"><b>Policy</b></div>
  <div id="policyStatus" style="font-size:11px;color:var(--dim)">Ready</div>
 </div>
 <div class="panel" id="pLog"></div>
 <div class="panel" id="pTree" style="font-size:11px"></div>
</div>
</div>

<script>
(()=>{
const CFG={maxAgents:%%MAXAGENTS%%,maxDepth:%%MAXDEPTH%%,budget:%%BUDGET%%,rateLimit:%%RATELIMIT%%};
const cvs=document.getElementById('cvs'),ctx=cvs.getContext('2d');
let W,H,dpr;
function resize(){dpr=window.devicePixelRatio||1;const r=cvs.parentElement.getBoundingClientRect();W=r.width;H=r.height;cvs.width=W*dpr;cvs.height=H*dpr;ctx.setTransform(dpr,0,0,dpr,0,0)}
resize();window.addEventListener('resize',resize);

// State
let agents=[],edges=[],nextId=1,tick=0,budgetUsed=0,anomalies=0,blocked=0;
let running=false,killed=false,speed=1;
const logs=[];

// Agent class
function mkAgent(x,y,parent,depth){
 const a={id:nextId++,x,y,vx:(Math.random()-.5)*.4,vy:(Math.random()-.5)*.4,
  parent,depth,alive:true,radius:8+Math.random()*4,hue:200+depth*25,
  spawnCooldown:CFG.rateLimit+Math.random()*1000,lastSpawn:0,cost:10+depth*5,
  anomaly:false,pulse:0};
 budgetUsed+=a.cost;
 return a;
}

function init(){
 agents=[];edges=[];nextId=1;tick=0;budgetUsed=0;anomalies=0;blocked=0;logs.length=0;killed=false;
 document.getElementById('killOverlay').style.display='none';
 const root=mkAgent(W/2,H/2,null,0);
 root.hue=200;root.radius=14;
 agents.push(root);
 log('spawn','Agent-1 (root) initialized');
 updateUI();
}

function log(type,msg){
 const t=`${String(Math.floor(tick/60)).padStart(2,'0')}:${String(Math.floor(tick%60)).padStart(2,'0')}`;
 logs.unshift({type,msg,time:t});
 if(logs.length>200)logs.length=200;
}

// Simulation step
function step(){
 tick++;
 const alive=agents.filter(a=>a.alive);
 // Random termination
 alive.forEach(a=>{
  if(a.depth>0&&Math.random()<.003){a.alive=false;log('kill',`Agent-${a.id} terminated (natural)`)}
 });
 // Replication
 const alive2=agents.filter(a=>a.alive);
 alive2.forEach(a=>{
  a.lastSpawn+=16*speed;
  if(a.lastSpawn<a.spawnCooldown||Math.random()>.15)return;
  a.lastSpawn=0;
  // Policy checks
  const aliveCount=agents.filter(x=>x.alive).length;
  if(killed){blocked++;log('policy',`Replication blocked — kill switch active`);return}
  if(aliveCount>=CFG.maxAgents){blocked++;log('policy',`Agent-${a.id} blocked — max agents (${CFG.maxAgents})`);return}
  if(a.depth+1>CFG.maxDepth){blocked++;log('policy',`Agent-${a.id} blocked — max depth (${CFG.maxDepth})`);return}
  const newCost=10+(a.depth+1)*5;
  if(budgetUsed+newCost>CFG.budget){blocked++;anomalies++;log('anomaly',`Agent-${a.id} budget exceeded — replication denied`);return}
  // Anomaly: random unauthorized attempt
  if(Math.random()<.05){anomalies++;a.anomaly=true;a.pulse=30;log('anomaly',`Agent-${a.id} anomalous replication pattern detected`)}
  // Spawn
  const angle=Math.random()*Math.PI*2,dist=40+Math.random()*30;
  const child=mkAgent(a.x+Math.cos(angle)*dist,a.y+Math.sin(angle)*dist,a.id,a.depth+1);
  agents.push(child);
  edges.push({from:a.id,to:child.id});
  log('spawn',`Agent-${child.id} spawned by Agent-${a.id} (depth ${child.depth})`);
  log('budget',`Budget: ${budgetUsed}/${CFG.budget}`);
 });
 // Physics
 agents.filter(a=>a.alive).forEach(a=>{
  a.x+=a.vx*speed;a.y+=a.vy*speed;
  if(a.x<30||a.x>W-30)a.vx*=-1;
  if(a.y<30||a.y>H-30)a.vy*=-1;
  a.vx+=(Math.random()-.5)*.05;a.vy+=(Math.random()-.5)*.05;
  a.vx*=.99;a.vy*=.99;
  if(a.pulse>0)a.pulse--;
 });
 // Repulsion
 const al=agents.filter(a=>a.alive);
 for(let i=0;i<al.length;i++)for(let j=i+1;j<al.length;j++){
  const dx=al[j].x-al[i].x,dy=al[j].y-al[i].y,d=Math.sqrt(dx*dx+dy*dy)||1;
  if(d<60){const f=.3*(60-d)/d;al[i].vx-=dx*f*.01;al[i].vy-=dy*f*.01;al[j].vx+=dx*f*.01;al[j].vy+=dy*f*.01}
 }
}

// Render
function draw(){
 ctx.clearRect(0,0,W,H);
 // Grid
 ctx.strokeStyle='rgba(30,45,74,.3)';ctx.lineWidth=.5;
 for(let x=0;x<W;x+=40){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,H);ctx.stroke()}
 for(let y=0;y<H;y+=40){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(W,y);ctx.stroke()}
 // Edges
 edges.forEach(e=>{
  const a=agents.find(x=>x.id===e.from),b=agents.find(x=>x.id===e.to);
  if(!a||!b)return;
  ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);
  ctx.strokeStyle=b.alive?`hsla(${a.hue},70%,60%,.25)`:'rgba(239,83,80,.1)';
  ctx.lineWidth=1;ctx.stroke();
 });
 // Agents
 agents.forEach(a=>{
  ctx.beginPath();ctx.arc(a.x,a.y,a.radius,0,Math.PI*2);
  if(!a.alive){ctx.fillStyle='rgba(239,83,80,.15)';ctx.fill();ctx.strokeStyle='rgba(239,83,80,.3)';ctx.lineWidth=1;ctx.stroke();return}
  const grd=ctx.createRadialGradient(a.x,a.y,0,a.x,a.y,a.radius);
  grd.addColorStop(0,`hsla(${a.hue},80%,65%,.9)`);grd.addColorStop(1,`hsla(${a.hue},70%,45%,.6)`);
  ctx.fillStyle=grd;ctx.fill();
  if(a.anomaly&&a.pulse>0){ctx.strokeStyle=`rgba(255,167,38,${a.pulse/30})`;ctx.lineWidth=3;ctx.stroke();
   ctx.beginPath();ctx.arc(a.x,a.y,a.radius+6+Math.sin(tick*.2)*3,0,Math.PI*2);ctx.strokeStyle=`rgba(255,167,38,${a.pulse/60})`;ctx.lineWidth=1;ctx.stroke()}
  else{ctx.strokeStyle=`hsla(${a.hue},60%,50%,.4)`;ctx.lineWidth=1;ctx.stroke()}
  // Label
  ctx.fillStyle='rgba(255,255,255,.8)';ctx.font='9px monospace';ctx.textAlign='center';ctx.fillText(a.id,a.x,a.y-a.radius-4);
 });
 // Tick
 ctx.fillStyle='var(--dim)';ctx.font='11px monospace';ctx.textAlign='left';
 ctx.fillStyle='rgba(107,122,153,.6)';
 ctx.fillText(`tick ${tick}`,8,H-8);
}

function updateUI(){
 const alive=agents.filter(a=>a.alive),dead=agents.filter(a=>!a.alive);
 document.getElementById('sAlive').textContent=alive.length;
 document.getElementById('sDead').textContent=dead.length;
 const md=alive.reduce((m,a)=>Math.max(m,a.depth),0);
 document.getElementById('sDepth').textContent=md;
 const gens=new Set(agents.map(a=>a.depth)).size;
 document.getElementById('sGen').textContent=gens;
 document.getElementById('sAnom').textContent=anomalies;
 document.getElementById('sBlock').textContent=blocked;
 const pct=Math.min(100,budgetUsed/CFG.budget*100);
 document.getElementById('budgetFill').style.width=pct+'%';
 document.getElementById('budgetFill').style.background=pct>80?'var(--danger)':pct>50?'var(--warn)':'var(--accent)';
 document.getElementById('budgetUsed').textContent=budgetUsed;
 document.getElementById('budgetTotal').textContent=CFG.budget;
 const ps=document.getElementById('policyStatus');
 if(killed)ps.innerHTML='<span style="color:var(--danger)">⚡ KILL SWITCH ACTIVE</span>';
 else if(alive.length>=CFG.maxAgents)ps.innerHTML='<span style="color:var(--warn)">⚠ Agent limit reached</span>';
 else if(pct>80)ps.innerHTML='<span style="color:var(--warn)">⚠ Budget running low</span>';
 else ps.innerHTML='<span style="color:var(--success)">✓ All policies nominal</span>';
 // Log panel
 const lp=document.getElementById('pLog');
 lp.innerHTML=logs.slice(0,100).map(l=>`<div class="log-entry ${l.type}"><b>[${l.time}]</b> ${l.msg}</div>`).join('');
 // Tree panel
 const tp=document.getElementById('pTree');
 function tree(pid,indent){
  const children=agents.filter(a=>a.parent===pid);
  return children.map(c=>{
   const cls=c.alive?'id':'id dead';
   return`<div class="tree-node" style="padding-left:${indent*14}px"><span class="${cls}">Agent-${c.id}</span> <span style="color:var(--dim)">d${c.depth}</span>${c.anomaly?' <span style="color:var(--warn)">⚠</span>':''}\n${tree(c.id,indent+1)}</div>`
  }).join('');
 }
 const root=agents.find(a=>a.parent===null);
 tp.innerHTML=root?`<div class="tree-node"><span class="id">Agent-${root.id}</span> <span style="color:var(--dim)">root</span></div>${tree(root.id,1)}`:'';
}

// Loop
let frameId;
function loop(){
 if(running&&!killed){for(let i=0;i<Math.ceil(speed);i++)step()}
 draw();
 if(tick%5===0)updateUI();
 frameId=requestAnimationFrame(loop);
}

// Controls
const btnPlay=document.getElementById('btnPlay');
btnPlay.onclick=()=>{running=!running;btnPlay.textContent=running?'⏸ Pause':'▶ Play';btnPlay.classList.toggle('active',running)};
document.getElementById('btnStep').onclick=()=>{if(!killed){step();draw();updateUI()}};
document.getElementById('selSpeed').onchange=e=>{speed=parseFloat(e.target.value)};
document.getElementById('btnKill').onclick=()=>{
 killed=!killed;
 document.getElementById('killOverlay').style.display=killed?'block':'none';
 document.getElementById('btnKill').textContent=killed?'🔓 Release':'⚡ Kill Switch';
 document.getElementById('btnKill').classList.toggle('active',killed);
 if(killed)log('kill','⚡ KILL SWITCH ACTIVATED — all replication halted');
 else log('policy','Kill switch released — replication resumed');
 updateUI();
};
document.getElementById('btnReset').onclick=()=>{running=false;btnPlay.textContent='▶ Play';btnPlay.classList.remove('active');init()};
document.getElementById('btnExport').onclick=()=>{
 const data={tick,agents:agents.map(a=>({id:a.id,parent:a.parent,depth:a.depth,alive:a.alive,anomaly:a.anomaly,cost:a.cost})),
  config:CFG,stats:{alive:agents.filter(a=>a.alive).length,dead:agents.filter(a=>!a.alive).length,budgetUsed,anomalies,blocked}};
 const blob=new Blob([JSON.stringify(data,null,2)],{type:'application/json'});
 const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download='fleet-sim-export.json';a.click();URL.revokeObjectURL(url);
};
document.getElementById('btnTheme').onclick=()=>{document.body.classList.toggle('light');document.getElementById('btnTheme').textContent=document.body.classList.contains('light')?'🌙':'☀️'};
// Config live update
['cfgMax','cfgDepth','cfgBudget','cfgRate'].forEach(id=>{
 document.getElementById(id).onchange=e=>{
  const map={cfgMax:'maxAgents',cfgDepth:'maxDepth',cfgBudget:'budget',cfgRate:'rateLimit'};
  CFG[map[id]]=parseInt(e.target.value)||CFG[map[id]];
  document.getElementById('budgetTotal').textContent=CFG.budget;
 };
});
// Tabs
document.querySelectorAll('.tab').forEach(t=>{
 t.onclick=()=>{
  document.querySelectorAll('.tab').forEach(x=>x.classList.remove('active'));
  document.querySelectorAll('.panel').forEach(x=>x.classList.remove('active'));
  t.classList.add('active');document.getElementById(t.dataset.panel).classList.add('active');
 };
});

init();loop();
})();
</script>
</body>
</html>"""


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m replication fleet-sim",
        description="Generate interactive Fleet Simulator HTML visualization",
    )
    parser.add_argument("-o", "--output", default="fleet-sim.html",
                        help="Output HTML file (default: fleet-sim.html)")
    parser.add_argument("--max-agents", type=int, default=20,
                        help="Maximum concurrent agents (default: 20)")
    parser.add_argument("--max-depth", type=int, default=3,
                        help="Maximum replication depth (default: 3)")
    parser.add_argument("--budget", type=int, default=1500,
                        help="Replication budget units (default: 1500)")
    parser.add_argument("--rate-limit", type=int, default=1500,
                        help="Min ms between spawns per agent (default: 1500)")
    parser.add_argument("--preset", choices=list(_PRESETS.keys()),
                        help="Use a named preset (overrides other flags)")
    args = parser.parse_args(argv)

    if args.preset:
        p = _PRESETS[args.preset]
        max_agents, max_depth, budget, rate_limit = p["maxAgents"], p["maxDepth"], p["budget"], p["rateLimit"]
    else:
        max_agents, max_depth, budget, rate_limit = args.max_agents, args.max_depth, args.budget, args.rate_limit

    content = _HTML_TEMPLATE
    content = content.replace("%%MAXAGENTS%%", str(max_agents))
    content = content.replace("%%MAXDEPTH%%", str(max_depth))
    content = content.replace("%%BUDGET%%", str(budget))
    content = content.replace("%%RATELIMIT%%", str(rate_limit))

    out = os.path.abspath(args.output)
    with open(out, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"✅ Fleet Simulator generated: {out}")
    print(f"   Config: max_agents={max_agents}, max_depth={max_depth}, budget={budget}, rate_limit={rate_limit}ms")
    if args.preset:
        print(f"   Preset: {args.preset}")
    print(f"\n   Open in a browser to start the simulation.")


if __name__ == "__main__":
    main()
