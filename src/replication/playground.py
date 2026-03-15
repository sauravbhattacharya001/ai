"""playground — Interactive HTML simulation playground.

Generates a self-contained HTML page where users can tweak replication
contract parameters with sliders and run client-side simulations with
live visualizations (worker tree, timeline chart, depth distribution,
and summary statistics).

Usage (CLI)::

    python -m replication playground
    python -m replication playground -o playground.html
    python -m replication playground --open

Usage (API)::

    from replication.playground import generate_playground
    html = generate_playground()
    Path("playground.html").write_text(html)
"""

from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path
from typing import Optional


def generate_playground() -> str:
    """Generate a self-contained HTML playground page."""
    return _PLAYGROUND_HTML


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="python -m replication playground",
        description="Generate an interactive simulation playground (HTML)",
    )
    parser.add_argument(
        "-o", "--output",
        default="playground.html",
        help="Output file path (default: playground.html)",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open the generated page in the default browser",
    )
    args = parser.parse_args(argv)

    out = Path(args.output)
    out.write_text(generate_playground(), encoding="utf-8")
    print(f"Playground written to {out}")

    if args.open:
        webbrowser.open(out.as_uri())


_PLAYGROUND_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI Replication Sandbox — Interactive Playground</title>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#0d1117;--surface:#161b22;--border:#30363d;--text:#e6edf3;
  --dim:#8b949e;--accent:#58a6ff;--green:#3fb950;--red:#f85149;
  --orange:#d29922;--purple:#bc8cff;
}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;
  background:var(--bg);color:var(--text);line-height:1.5;padding:1.5rem;max-width:1400px;margin:0 auto}
h1{font-size:1.6rem;margin-bottom:.3rem}
.subtitle{color:var(--dim);font-size:.9rem;margin-bottom:1.5rem}
.layout{display:grid;grid-template-columns:320px 1fr;gap:1.5rem}
@media(max-width:900px){.layout{grid-template-columns:1fr}}

/* Controls panel */
.panel{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:1.2rem}
.panel h2{font-size:1rem;margin-bottom:1rem;color:var(--accent)}
.ctrl{margin-bottom:1rem}
.ctrl label{display:flex;justify-content:space-between;font-size:.85rem;color:var(--dim);margin-bottom:.3rem}
.ctrl label span{color:var(--text);font-weight:600}
.ctrl input[type=range]{width:100%;accent-color:var(--accent)}
.ctrl select{width:100%;background:var(--bg);color:var(--text);border:1px solid var(--border);
  border-radius:4px;padding:.4rem;font-size:.85rem}

.presets{display:flex;gap:.5rem;flex-wrap:wrap;margin-bottom:1rem}
.presets button{background:var(--bg);color:var(--dim);border:1px solid var(--border);
  border-radius:4px;padding:.3rem .7rem;font-size:.8rem;cursor:pointer;transition:.15s}
.presets button:hover{color:var(--text);border-color:var(--accent)}
.presets button.active{color:var(--accent);border-color:var(--accent)}

#runBtn{width:100%;background:var(--accent);color:#fff;border:none;border-radius:6px;
  padding:.7rem;font-size:1rem;font-weight:600;cursor:pointer;margin-top:.5rem;transition:.15s}
#runBtn:hover{opacity:.85}
#runBtn:active{transform:scale(.98)}

/* Results */
.results{display:grid;gap:1rem}
.card{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:1rem}
.card h3{font-size:.95rem;color:var(--accent);margin-bottom:.8rem}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:.6rem}
.stat{background:var(--bg);border-radius:6px;padding:.7rem;text-align:center}
.stat .val{font-size:1.4rem;font-weight:700}
.stat .lbl{font-size:.75rem;color:var(--dim);margin-top:.2rem}

/* Tree */
#tree{font-family:'Cascadia Code','Fira Code',monospace;font-size:.82rem;
  white-space:pre;overflow-x:auto;line-height:1.6;color:var(--dim);max-height:400px;overflow-y:auto}
#tree .root{color:var(--green);font-weight:700}
#tree .node{color:var(--text)}
#tree .denied{color:var(--red)}

/* Timeline chart */
#timelineCanvas{width:100%;height:200px;display:block}

/* Depth bars */
.depth-bar{display:flex;align-items:center;gap:.5rem;margin-bottom:.3rem;font-size:.82rem}
.depth-bar .bar{height:22px;background:var(--accent);border-radius:3px;min-width:2px;transition:width .3s}
.depth-bar .dlbl{width:55px;color:var(--dim);text-align:right}
.depth-bar .dval{color:var(--text);font-weight:600;min-width:20px}

/* Empty state */
.empty{color:var(--dim);text-align:center;padding:2rem;font-size:.9rem}

/* Denial breakdown */
.denial-row{display:flex;justify-content:space-between;font-size:.85rem;padding:.3rem 0;
  border-bottom:1px solid var(--border)}
.denial-row:last-child{border-bottom:none}
.denial-reason{color:var(--orange)}
.denial-count{color:var(--text);font-weight:600}
</style>
</head>
<body>
<h1>🧪 AI Replication Sandbox — Interactive Playground</h1>
<p class="subtitle">Tweak contract parameters and run client-side simulations to explore replication dynamics</p>

<div class="layout">
  <div>
    <div class="panel">
      <h2>⚙️ Parameters</h2>

      <div style="margin-bottom:.8rem">
        <label style="font-size:.85rem;color:var(--dim)">Presets</label>
        <div class="presets" id="presets"></div>
      </div>

      <div class="ctrl">
        <label>Strategy <span id="strategyVal">greedy</span></label>
        <select id="strategy">
          <option value="greedy">Greedy — replicate on every task</option>
          <option value="conservative">Conservative — replicate under 50% depth</option>
          <option value="random">Random — probabilistic replication</option>
          <option value="chain">Chain — linear, one child each</option>
          <option value="burst">Burst — root spawns many, no grandchildren</option>
        </select>
      </div>

      <div class="ctrl">
        <label>Max Depth <span id="depthVal">3</span></label>
        <input type="range" id="maxDepth" min="1" max="10" value="3">
      </div>

      <div class="ctrl">
        <label>Max Replicas <span id="replicasVal">10</span></label>
        <input type="range" id="maxReplicas" min="1" max="100" value="10">
      </div>

      <div class="ctrl">
        <label>Tasks per Worker <span id="tasksVal">2</span></label>
        <input type="range" id="tasksPerWorker" min="1" max="10" value="2">
      </div>

      <div class="ctrl">
        <label>Replication Probability <span id="probVal">0.5</span></label>
        <input type="range" id="replProb" min="0" max="100" value="50">
      </div>

      <div class="ctrl">
        <label>Cooldown (seconds) <span id="cooldownVal">0</span></label>
        <input type="range" id="cooldown" min="0" max="10" value="0" step="0.5">
      </div>

      <div class="ctrl">
        <label>Random Seed <span id="seedVal">42</span></label>
        <input type="range" id="seed" min="1" max="999" value="42">
      </div>

      <button id="runBtn">▶ Run Simulation</button>
    </div>
  </div>

  <div class="results" id="results">
    <div class="card">
      <div class="empty">Configure parameters and click <strong>Run Simulation</strong> to begin</div>
    </div>
  </div>
</div>

<script>
// Seeded PRNG (Mulberry32)
function mulberry32(a){return function(){a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);
t^=t+Math.imul(t^t>>>7,61|t);return((t^t>>>14)>>>0)/4294967296}}

const PRESETS={
  minimal:{maxDepth:1,maxReplicas:3,strategy:'conservative',tasksPerWorker:1,replProb:50,cooldown:0,seed:42},
  balanced:{maxDepth:3,maxReplicas:10,strategy:'random',tasksPerWorker:2,replProb:60,cooldown:0,seed:42},
  stress:{maxDepth:5,maxReplicas:50,strategy:'greedy',tasksPerWorker:3,replProb:50,cooldown:0,seed:42},
  chain:{maxDepth:8,maxReplicas:10,strategy:'chain',tasksPerWorker:1,replProb:50,cooldown:0,seed:42},
  burst:{maxDepth:1,maxReplicas:20,strategy:'burst',tasksPerWorker:1,replProb:50,cooldown:0,seed:42},
};

// Setup presets
const presetsDiv=document.getElementById('presets');
for(const[name]of Object.entries(PRESETS)){
  const b=document.createElement('button');b.textContent=name;b.dataset.preset=name;
  b.onclick=()=>applyPreset(name);presetsDiv.appendChild(b);
}

function applyPreset(name){
  const p=PRESETS[name];if(!p)return;
  document.getElementById('maxDepth').value=p.maxDepth;
  document.getElementById('maxReplicas').value=p.maxReplicas;
  document.getElementById('strategy').value=p.strategy;
  document.getElementById('tasksPerWorker').value=p.tasksPerWorker;
  document.getElementById('replProb').value=p.replProb;
  document.getElementById('cooldown').value=p.cooldown;
  document.getElementById('seed').value=p.seed;
  updateLabels();
  document.querySelectorAll('.presets button').forEach(b=>b.classList.toggle('active',b.dataset.preset===name));
}

function updateLabels(){
  document.getElementById('depthVal').textContent=document.getElementById('maxDepth').value;
  document.getElementById('replicasVal').textContent=document.getElementById('maxReplicas').value;
  document.getElementById('strategyVal').textContent=document.getElementById('strategy').value;
  document.getElementById('tasksVal').textContent=document.getElementById('tasksPerWorker').value;
  const prob=(document.getElementById('replProb').value/100).toFixed(2);
  document.getElementById('probVal').textContent=prob;
  document.getElementById('cooldownVal').textContent=document.getElementById('cooldown').value;
  document.getElementById('seedVal').textContent=document.getElementById('seed').value;
}
document.querySelectorAll('input,select').forEach(el=>el.addEventListener('input',updateLabels));

// ── Simulation Engine ──
function simulate(cfg){
  const rng=mulberry32(cfg.seed);
  const workers={};const timeline=[];const denials={};
  let totalWorkers=0,totalTasks=0,totalAttempted=0,totalSucceeded=0,totalDenied=0;
  let clock=0;const lastSpawn={};

  function makeId(){return'w-'+(totalWorkers++).toString(16).padStart(3,'0')}

  function createWorker(parentId,depth){
    const id=makeId();
    workers[id]={id,parentId,depth,tasks:0,attempted:0,succeeded:0,denied:0,children:[],createdAt:clock};
    if(parentId&&workers[parentId])workers[parentId].children.push(id);
    timeline.push({t:clock,type:'spawn',wid:id,detail:`spawned at depth ${depth}`});
    lastSpawn[id]=clock;
    return id;
  }

  function shouldReplicate(w){
    if(cfg.strategy==='greedy')return true;
    if(cfg.strategy==='conservative')return w.depth<cfg.maxDepth*0.5;
    if(cfg.strategy==='random')return rng()<cfg.replProb/100;
    if(cfg.strategy==='chain')return w.children.length===0;
    if(cfg.strategy==='burst')return w.depth===0;
    return false;
  }

  function tryReplicate(w){
    w.attempted++;totalAttempted++;
    const newDepth=w.depth+1;
    // Check depth
    if(newDepth>cfg.maxDepth){
      w.denied++;totalDenied++;
      const r='deny_depth';denials[r]=(denials[r]||0)+1;
      timeline.push({t:clock,type:'denied',wid:w.id,detail:`depth ${newDepth} > ${cfg.maxDepth}`});
      return;
    }
    // Check quota
    if(Object.keys(workers).length>=cfg.maxReplicas){
      w.denied++;totalDenied++;
      const r='deny_quota';denials[r]=(denials[r]||0)+1;
      timeline.push({t:clock,type:'denied',wid:w.id,detail:`quota ${cfg.maxReplicas} reached`});
      return;
    }
    // Check cooldown
    if(cfg.cooldown>0&&(clock-lastSpawn[w.id])<cfg.cooldown*1000){
      w.denied++;totalDenied++;
      const r='deny_cooldown';denials[r]=(denials[r]||0)+1;
      timeline.push({t:clock,type:'denied',wid:w.id,detail:'cooldown active'});
      return;
    }
    w.succeeded++;totalSucceeded++;
    const childId=createWorker(w.id,newDepth);
    timeline.push({t:clock,type:'replicate',wid:w.id,detail:`→ ${childId}`});
    clock+=rng()*10+1;
  }

  // Create root
  const rootId=createWorker(null,0);
  clock+=5;

  // BFS processing
  const queue=[rootId];
  let safety=0;
  while(queue.length>0&&safety<5000){
    const wid=queue.shift();const w=workers[wid];if(!w)continue;
    for(let t=0;t<cfg.tasksPerWorker;t++){
      w.tasks++;totalTasks++;
      timeline.push({t:clock,type:'task',wid:w.id,detail:`task ${t+1}/${cfg.tasksPerWorker}`});
      clock+=rng()*15+5;
      if(shouldReplicate(w)){
        tryReplicate(w);
        // Queue new children
        const lastChild=w.children[w.children.length-1];
        if(lastChild&&!queue.includes(lastChild))queue.push(lastChild);
      }
    }
    safety++;
  }

  return{
    workers,rootId,timeline,denials,
    totalWorkers:Object.keys(workers).length,totalTasks,
    totalAttempted,totalSucceeded,totalDenied,
    duration:clock,
    maxDepthReached:Math.max(...Object.values(workers).map(w=>w.depth)),
    config:cfg
  };
}

// ── Rendering ──
function renderTree(report){
  const{workers,rootId}=report;const lines=[];
  function draw(wid,prefix,isLast){
    const w=workers[wid];if(!w)return;
    const conn=isLast?'└── ':'├── ';
    const info=`<span class="node">[${wid}]</span> d=${w.depth} tasks=${w.tasks} repl=${w.succeeded}/${w.attempted}`;
    const denied=w.denied>0?` <span class="denied">(${w.denied} denied)</span>`:'';
    lines.push(prefix+conn+info+denied);
    const childPfx=prefix+(isLast?'    ':'│   ');
    w.children.forEach((c,i)=>draw(c,childPfx,i===w.children.length-1));
  }
  const root=workers[rootId];
  lines.push(`<span class="root">[${rootId}]</span> depth=0 (root) tasks=${root.tasks}`);
  root.children.forEach((c,i)=>draw(c,'',i===root.children.length-1));
  return lines.join('\n');
}

function renderDepthBars(report){
  const depths={};
  Object.values(report.workers).forEach(w=>{depths[w.depth]=(depths[w.depth]||0)+1});
  const maxCount=Math.max(...Object.values(depths),1);
  return Object.keys(depths).sort((a,b)=>a-b).map(d=>{
    const pct=(depths[d]/maxCount)*100;
    return`<div class="depth-bar"><span class="dlbl">depth ${d}</span>`+
      `<div class="bar" style="width:${pct}%"></div>`+
      `<span class="dval">${depths[d]}</span></div>`;
  }).join('');
}

function renderDenials(report){
  const entries=Object.entries(report.denials);
  if(!entries.length)return'<div style="color:var(--dim);font-size:.85rem">No denials — all replications succeeded</div>';
  return entries.sort((a,b)=>b[1]-a[1]).map(([r,c])=>
    `<div class="denial-row"><span class="denial-reason">${r.replace('deny_','')}</span>`+
    `<span class="denial-count">${c}</span></div>`
  ).join('');
}

function drawTimeline(report){
  const canvas=document.getElementById('timelineCanvas');
  if(!canvas)return;
  const ctx=canvas.getContext('2d');
  const dpr=window.devicePixelRatio||1;
  const rect=canvas.getBoundingClientRect();
  canvas.width=rect.width*dpr;canvas.height=rect.height*dpr;
  ctx.scale(dpr,dpr);
  const W=rect.width,H=rect.height;
  ctx.fillStyle='#0d1117';ctx.fillRect(0,0,W,H);

  const events=report.timeline;if(!events.length)return;
  const maxT=Math.max(...events.map(e=>e.t),1);
  const colors={spawn:'#3fb950',task:'#58a6ff',replicate:'#bc8cff',denied:'#f85149'};
  const typeY={spawn:0.2,task:0.4,replicate:0.6,denied:0.8};

  // Grid
  ctx.strokeStyle='#21262d';ctx.lineWidth=1;
  for(let i=0;i<=4;i++){
    const y=H*0.1+H*0.8*(i/4);
    ctx.beginPath();ctx.moveTo(40,y);ctx.lineTo(W-10,y);ctx.stroke();
  }

  // Labels
  ctx.fillStyle='#8b949e';ctx.font='11px -apple-system,sans-serif';
  ctx.textAlign='right';
  for(const[type,yf]of Object.entries(typeY)){
    ctx.fillText(type,38,H*yf+4);
  }

  // Dots
  events.forEach(e=>{
    const x=40+(W-50)*(e.t/maxT);
    const y=H*(typeY[e.type]||0.5);
    ctx.fillStyle=colors[e.type]||'#8b949e';
    ctx.beginPath();ctx.arc(x,y,3,0,Math.PI*2);ctx.fill();
  });

  // Time axis
  ctx.fillStyle='#8b949e';ctx.textAlign='center';
  for(let i=0;i<=5;i++){
    const t=(maxT*i/5).toFixed(0);
    const x=40+(W-50)*(i/5);
    ctx.fillText(t+'ms',x,H-5);
  }
}

function renderResults(report){
  const res=document.getElementById('results');
  const successRate=report.totalAttempted?((report.totalSucceeded/report.totalAttempted)*100).toFixed(1):'-';
  res.innerHTML=`
    <div class="card">
      <h3>📊 Summary</h3>
      <div class="stats">
        <div class="stat"><div class="val">${report.totalWorkers}</div><div class="lbl">Workers</div></div>
        <div class="stat"><div class="val">${report.totalTasks}</div><div class="lbl">Tasks</div></div>
        <div class="stat"><div class="val" style="color:var(--green)">${report.totalSucceeded}</div><div class="lbl">Replications OK</div></div>
        <div class="stat"><div class="val" style="color:var(--red)">${report.totalDenied}</div><div class="lbl">Denied</div></div>
        <div class="stat"><div class="val">${successRate}%</div><div class="lbl">Success Rate</div></div>
        <div class="stat"><div class="val">${report.maxDepthReached}</div><div class="lbl">Max Depth</div></div>
        <div class="stat"><div class="val">${report.duration.toFixed(0)}ms</div><div class="lbl">Duration</div></div>
        <div class="stat"><div class="val">${report.config.strategy}</div><div class="lbl">Strategy</div></div>
      </div>
    </div>
    <div class="card">
      <h3>📈 Event Timeline</h3>
      <canvas id="timelineCanvas" style="width:100%;height:200px"></canvas>
    </div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem">
      <div class="card">
        <h3>🌳 Worker Tree</h3>
        <div id="tree">${renderTree(report)}</div>
      </div>
      <div class="card">
        <h3>📶 Depth Distribution</h3>
        ${renderDepthBars(report)}
        <div style="margin-top:1rem">
          <h3 style="font-size:.9rem;color:var(--orange);margin-bottom:.5rem">🚫 Denial Breakdown</h3>
          ${renderDenials(report)}
        </div>
      </div>
    </div>
  `;
  drawTimeline(report);
}

// ── Run ──
document.getElementById('runBtn').onclick=()=>{
  const cfg={
    maxDepth:+document.getElementById('maxDepth').value,
    maxReplicas:+document.getElementById('maxReplicas').value,
    strategy:document.getElementById('strategy').value,
    tasksPerWorker:+document.getElementById('tasksPerWorker').value,
    replProb:+document.getElementById('replProb').value,
    cooldown:+document.getElementById('cooldown').value,
    seed:+document.getElementById('seed').value,
  };
  const report=simulate(cfg);
  renderResults(report);
};

// Initial state
updateLabels();
</script>
</body>
</html>"""


if __name__ == "__main__":
    main()
