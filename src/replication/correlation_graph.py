"""correlation_graph — Interactive HTML visualization of threat correlation networks.

Generates a self-contained HTML page with a force-directed graph where
nodes represent detection signals and edges represent correlations between
them.  Supports filtering by source, severity, and agent, plus a timeline
slider to explore temporal patterns.

Features:

* **Force-directed layout** — signals as nodes, correlations as edges
* **Color coding** — node color by severity (info→critical gradient)
* **Node sizing** — by signal severity weight
* **Edge styling** — dashed for temporal, solid for same-agent
* **Source filters** — toggle signal sources on/off
* **Severity filters** — toggle severity levels
* **Agent filter** — dropdown to focus on a single agent
* **Timeline slider** — scrub through time to see signal progression
* **Detail panel** — click a node/edge for full details
* **Cluster highlighting** — highlight compound threat clusters
* **PNG export** — save current view as image
* **Demo data** — generates realistic sample data for exploration

Usage (CLI)::

    python -m replication correlation-graph
    python -m replication correlation-graph -o graph.html
    python -m replication correlation-graph --open

Usage (API)::

    from replication.correlation_graph import generate_correlation_graph
    html = generate_correlation_graph()
    Path("graph.html").write_text(html)
"""

from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path
from typing import Optional


def generate_correlation_graph() -> str:
    """Generate a self-contained HTML correlation graph page."""
    return _CORRELATION_GRAPH_HTML


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="python -m replication correlation-graph",
        description="Generate an interactive threat correlation graph (HTML)",
    )
    parser.add_argument(
        "-o", "--output",
        default="correlation_graph.html",
        help="Output file path (default: correlation_graph.html)",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open the generated page in the default browser",
    )
    args = parser.parse_args(argv)
    html = generate_correlation_graph()
    Path(args.output).write_text(html, encoding="utf-8")
    print(f"Correlation graph written to {args.output}")
    if args.open:
        webbrowser.open(str(Path(args.output).resolve()))


_CORRELATION_GRAPH_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Threat Correlation Graph</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#0a0e17;color:#e0e0e0;overflow:hidden;height:100vh;display:flex;flex-direction:column}
header{background:#141929;padding:12px 20px;display:flex;align-items:center;gap:16px;border-bottom:1px solid #1e2740;flex-shrink:0}
header h1{font-size:18px;color:#7eb8ff;white-space:nowrap}
.toolbar{display:flex;gap:8px;flex-wrap:wrap;align-items:center;flex:1}
.toolbar button,.toolbar select{background:#1e2740;color:#c0d0e8;border:1px solid #2a3a5c;border-radius:6px;padding:5px 12px;font-size:13px;cursor:pointer;transition:all .15s}
.toolbar button:hover,.toolbar select:hover{background:#2a3a5c;border-color:#4a6a9c}
.toolbar button.active{background:#2a5a8c;border-color:#4a8adc}
.filter-group{display:flex;gap:4px;align-items:center}
.filter-group label{font-size:12px;color:#8090a8;margin-right:4px}
.main{display:flex;flex:1;overflow:hidden}
canvas{flex:1;display:block}
.sidebar{width:300px;background:#141929;border-left:1px solid #1e2740;padding:16px;overflow-y:auto;flex-shrink:0;font-size:13px;display:none}
.sidebar.open{display:block}
.sidebar h2{font-size:15px;color:#7eb8ff;margin-bottom:12px}
.sidebar .field{margin-bottom:8px}
.sidebar .field-label{color:#8090a8;font-size:11px;text-transform:uppercase;margin-bottom:2px}
.sidebar .field-value{color:#e0e0e0}
.timeline{background:#141929;border-top:1px solid #1e2740;padding:10px 20px;flex-shrink:0;display:flex;align-items:center;gap:12px}
.timeline label{font-size:12px;color:#8090a8;white-space:nowrap}
.timeline input[type=range]{flex:1;accent-color:#4a8adc}
.timeline .time-display{font-size:12px;color:#c0d0e8;min-width:100px;text-align:right}
.severity-badge{display:inline-block;padding:2px 8px;border-radius:10px;font-size:11px;font-weight:600}
.sev-info{background:#1a3a2a;color:#4ade80}
.sev-low{background:#1a3a4a;color:#60c0f0}
.sev-medium{background:#3a3a1a;color:#f0d060}
.sev-high{background:#3a2a1a;color:#f0a040}
.sev-critical{background:#3a1a1a;color:#f06060}
.legend{position:absolute;bottom:60px;left:16px;background:rgba(20,25,41,.9);border:1px solid #1e2740;border-radius:8px;padding:12px;font-size:12px}
.legend h3{font-size:13px;color:#7eb8ff;margin-bottom:8px}
.legend-item{display:flex;align-items:center;gap:8px;margin-bottom:4px}
.legend-dot{width:12px;height:12px;border-radius:50%}
.stats{position:absolute;top:60px;right:16px;background:rgba(20,25,41,.9);border:1px solid #1e2740;border-radius:8px;padding:12px;font-size:12px}
.stats h3{font-size:13px;color:#7eb8ff;margin-bottom:8px}
.stat-row{display:flex;justify-content:space-between;gap:16px;margin-bottom:4px}
.stat-val{color:#7eb8ff;font-weight:600}
.cluster-badge{display:inline-block;padding:1px 6px;border-radius:8px;font-size:10px;background:#2a3a5c;color:#a0b8d8;margin-left:4px}
</style>
</head>
<body>
<header>
<h1>🕸️ Threat Correlation Graph</h1>
<div class="toolbar">
<div class="filter-group">
<label>Source:</label>
<button class="src-btn active" data-src="all">All</button>
<button class="src-btn" data-src="drift">Drift</button>
<button class="src-btn" data-src="escalation">Escalation</button>
<button class="src-btn" data-src="killchain">Kill Chain</button>
<button class="src-btn" data-src="canary">Canary</button>
<button class="src-btn" data-src="covert">Covert</button>
<button class="src-btn" data-src="behavior">Behavior</button>
</div>
<div class="filter-group">
<label>Severity:</label>
<button class="sev-btn active" data-sev="all">All</button>
<button class="sev-btn" data-sev="critical">Critical</button>
<button class="sev-btn" data-sev="high">High</button>
<button class="sev-btn" data-sev="medium">Med</button>
<button class="sev-btn" data-sev="low">Low</button>
</div>
<div class="filter-group">
<label>Agent:</label>
<select id="agentFilter"><option value="all">All Agents</option></select>
</div>
<button id="exportBtn" title="Export PNG">📸 Export</button>
<button id="resetBtn" title="Reset view">🔄 Reset</button>
</div>
</header>
<div class="main">
<canvas id="graph"></canvas>
<div class="sidebar" id="sidebar">
<h2 id="sidebarTitle">Details</h2>
<div id="sidebarContent"></div>
</div>
</div>
<div class="timeline">
<label>Timeline:</label>
<input type="range" id="timeSlider" min="0" max="1000" value="1000">
<div class="time-display" id="timeDisplay">All events</div>
</div>

<div class="legend">
<h3>Severity</h3>
<div class="legend-item"><div class="legend-dot" style="background:#f06060"></div>Critical</div>
<div class="legend-item"><div class="legend-dot" style="background:#f0a040"></div>High</div>
<div class="legend-item"><div class="legend-dot" style="background:#f0d060"></div>Medium</div>
<div class="legend-item"><div class="legend-dot" style="background:#60c0f0"></div>Low</div>
<div class="legend-item"><div class="legend-dot" style="background:#4ade80"></div>Info</div>
<h3 style="margin-top:8px">Edges</h3>
<div class="legend-item"><div style="width:20px;height:2px;background:#4a8adc"></div>Same agent</div>
<div class="legend-item"><div style="width:20px;height:2px;background:#4a8adc;border-top:2px dashed #4a8adc"></div>Cross agent</div>
</div>

<div class="stats" id="statsPanel">
<h3>Graph Stats</h3>
<div class="stat-row"><span>Signals</span><span class="stat-val" id="statNodes">0</span></div>
<div class="stat-row"><span>Correlations</span><span class="stat-val" id="statEdges">0</span></div>
<div class="stat-row"><span>Clusters</span><span class="stat-val" id="statClusters">0</span></div>
<div class="stat-row"><span>Agents</span><span class="stat-val" id="statAgents">0</span></div>
<div class="stat-row"><span>Avg Severity</span><span class="stat-val" id="statAvgSev">0</span></div>
</div>

<script>
(function(){
"use strict";

// ── Demo Data Generation ──
const SOURCES=["drift","escalation","killchain","canary","covert","behavior"];
const SEVERITIES=["info","low","medium","high","critical"];
const SEV_WEIGHT={info:1,low:2,medium:3,high:4,critical:5};
const SEV_COLOR={info:"#4ade80",low:"#60c0f0",medium:"#f0d060",high:"#f0a040",critical:"#f06060"};
const AGENTS=["agent-alpha","agent-beta","agent-gamma","agent-delta","agent-epsilon"];
const DESCRIPTIONS={
  drift:["escape_rate spike z=%.1f","mutation_rate anomaly","parameter drift detected","behavioral divergence"],
  escalation:["privilege escalation attempt","unauthorized resource access","scope expansion detected","permission boundary probe"],
  killchain:["reconnaissance phase detected","lateral movement observed","payload delivery attempt","persistence mechanism found"],
  canary:["canary token triggered","trap file accessed","honeypot interaction","decoy endpoint hit"],
  covert:["steganographic encoding detected","timing channel activity","covert data exfiltration","hidden communication pattern"],
  behavior:["anomalous tool usage","unusual scheduling pattern","resource hoarding behavior","goal misalignment signal"]
};

function rng(seed){let s=seed;return()=>{s=(s*16807+0)%2147483647;return(s-1)/2147483646}}
function pick(arr,r){return arr[Math.floor(r()*arr.length)]}
function fmt(t,r){return t.replace("%.1f",(r()*5+1).toFixed(1))}

function generateData(){
  const r=rng(42);
  const signals=[];
  const edges=[];
  const N=60;
  for(let i=0;i<N;i++){
    const src=pick(SOURCES,r);
    const sev=pick(SEVERITIES,r);
    const agent=pick(AGENTS,r);
    const desc=fmt(pick(DESCRIPTIONS[src],r),r);
    const t=Math.floor(r()*1000);
    signals.push({id:i,source:src,severity:sev,agent,description:desc,timestamp:t,
      x:r()*800+100,y:r()*500+50,vx:0,vy:0});
  }
  // Generate correlations: signals from same agent within time window
  for(let i=0;i<N;i++){
    for(let j=i+1;j<N;j++){
      const si=signals[i],sj=signals[j];
      const timeDist=Math.abs(si.timestamp-sj.timestamp);
      if(timeDist<80){
        const sameAgent=si.agent===sj.agent;
        const prob=sameAgent?0.4:0.08;
        if(r()<prob){
          const combined=(SEV_WEIGHT[si.severity]+SEV_WEIGHT[sj.severity])/2;
          edges.push({source:i,target:j,sameAgent,strength:combined/5,
            rule:sameAgent?"temporal_correlation":"cross_agent_correlation"});
        }
      }
    }
  }
  // Detect clusters via connected components
  const parent=signals.map((_,i)=>i);
  function find(x){return parent[x]===x?x:parent[x]=find(parent[x])}
  function union(a,b){parent[find(a)]=find(b)}
  edges.forEach(e=>union(e.source,e.target));
  const clusters={};
  signals.forEach((s,i)=>{const c=find(i);if(!clusters[c])clusters[c]=[];clusters[c].push(i);s.cluster=c});
  const clusterIds=[...new Set(Object.keys(clusters))];
  const CLUSTER_COLORS=["#4a8adc","#dc4a8a","#8adc4a","#dc8a4a","#4adc8a","#8a4adc","#dcdc4a","#4adcdc"];
  signals.forEach(s=>{s.clusterColor=CLUSTER_COLORS[clusterIds.indexOf(String(s.cluster))%CLUSTER_COLORS.length]});
  return{signals,edges,clusters:Object.values(clusters).filter(c=>c.length>1)};
}

const data=generateData();
let filteredSignals=[...data.signals];
let filteredEdges=[...data.edges];
let selectedNode=null;
let selectedEdge=null;
let activeSource="all";
let activeSev="all";
let activeAgent="all";
let timeMax=1000;
let dragNode=null;
let panX=0,panY=0,zoom=1;
let isDragging=false,lastMouse={x:0,y:0};

// Populate agent filter
const agentSel=document.getElementById("agentFilter");
const agents=[...new Set(data.signals.map(s=>s.agent))].sort();
agents.forEach(a=>{const o=document.createElement("option");o.value=a;o.textContent=a;agentSel.appendChild(o)});

// Canvas setup
const canvas=document.getElementById("graph");
const ctx=canvas.getContext("2d");
function resize(){canvas.width=canvas.parentElement.clientWidth;canvas.height=canvas.parentElement.clientHeight}
window.addEventListener("resize",resize);resize();

// ── Filtering ──
function applyFilters(){
  filteredSignals=data.signals.filter(s=>{
    if(activeSource!=="all"&&s.source!==activeSource)return false;
    if(activeSev!=="all"&&s.severity!==activeSev)return false;
    if(activeAgent!=="all"&&s.agent!==activeAgent)return false;
    if(s.timestamp>timeMax)return false;
    return true;
  });
  const ids=new Set(filteredSignals.map(s=>s.id));
  filteredEdges=data.edges.filter(e=>ids.has(e.source)&&ids.has(e.target));
  updateStats();
}

function updateStats(){
  document.getElementById("statNodes").textContent=filteredSignals.length;
  document.getElementById("statEdges").textContent=filteredEdges.length;
  const clusterSet=new Set();
  filteredSignals.forEach(s=>clusterSet.add(s.cluster));
  document.getElementById("statClusters").textContent=data.clusters.filter(c=>c.some(id=>filteredSignals.find(s=>s.id===id))).length;
  document.getElementById("statAgents").textContent=new Set(filteredSignals.map(s=>s.agent)).size;
  const avg=filteredSignals.length?filteredSignals.reduce((a,s)=>a+SEV_WEIGHT[s.severity],0)/filteredSignals.length:0;
  document.getElementById("statAvgSev").textContent=avg.toFixed(1);
}

// ── Source filter buttons ──
document.querySelectorAll(".src-btn").forEach(btn=>{
  btn.addEventListener("click",()=>{
    document.querySelectorAll(".src-btn").forEach(b=>b.classList.remove("active"));
    btn.classList.add("active");
    activeSource=btn.dataset.src;
    applyFilters();
  });
});
document.querySelectorAll(".sev-btn").forEach(btn=>{
  btn.addEventListener("click",()=>{
    document.querySelectorAll(".sev-btn").forEach(b=>b.classList.remove("active"));
    btn.classList.add("active");
    activeSev=btn.dataset.sev;
    applyFilters();
  });
});
agentSel.addEventListener("change",()=>{activeAgent=agentSel.value;applyFilters()});

// Timeline
const timeSlider=document.getElementById("timeSlider");
const timeDisplay=document.getElementById("timeDisplay");
timeSlider.addEventListener("input",()=>{
  timeMax=parseInt(timeSlider.value);
  timeDisplay.textContent=timeMax>=1000?"All events":"t ≤ "+timeMax;
  applyFilters();
});

// ── Physics simulation ──
function simulate(){
  const nodes=filteredSignals;
  const edges=filteredEdges;
  const nodeMap={};
  nodes.forEach(n=>{nodeMap[n.id]=n});

  // Repulsion
  for(let i=0;i<nodes.length;i++){
    for(let j=i+1;j<nodes.length;j++){
      const a=nodes[i],b=nodes[j];
      let dx=b.x-a.x,dy=b.y-a.y;
      let d=Math.sqrt(dx*dx+dy*dy)||1;
      if(d<200){
        const f=800/(d*d);
        const fx=dx/d*f,fy=dy/d*f;
        a.vx-=fx;a.vy-=fy;b.vx+=fx;b.vy+=fy;
      }
    }
  }
  // Attraction along edges
  edges.forEach(e=>{
    const a=nodeMap[e.source],b=nodeMap[e.target];
    if(!a||!b)return;
    let dx=b.x-a.x,dy=b.y-a.y;
    let d=Math.sqrt(dx*dx+dy*dy)||1;
    const f=(d-120)*0.005;
    const fx=dx/d*f,fy=dy/d*f;
    a.vx+=fx;a.vy+=fy;b.vx-=fx;b.vy-=fy;
  });
  // Center gravity
  const cx=canvas.width/2,cy=canvas.height/2;
  nodes.forEach(n=>{
    n.vx+=(cx-n.x)*0.0005;
    n.vy+=(cy-n.y)*0.0005;
    n.vx*=0.9;n.vy*=0.9;
    if(n!==dragNode){n.x+=n.vx;n.y+=n.vy}
    n.x=Math.max(30,Math.min(canvas.width-30,n.x));
    n.y=Math.max(30,Math.min(canvas.height-30,n.y));
  });
}

// ── Rendering ──
function draw(){
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.save();
  ctx.translate(panX,panY);
  ctx.scale(zoom,zoom);

  const nodeMap={};
  filteredSignals.forEach(n=>{nodeMap[n.id]=n});

  // Draw edges
  filteredEdges.forEach(e=>{
    const a=nodeMap[e.source],b=nodeMap[e.target];
    if(!a||!b)return;
    ctx.beginPath();
    ctx.strokeStyle=selectedEdge===e?"#fff":"rgba(74,138,220,"+(0.2+e.strength*0.4)+")";
    ctx.lineWidth=selectedEdge===e?3:1+e.strength*2;
    if(!e.sameAgent){ctx.setLineDash([4,4])}else{ctx.setLineDash([])}
    ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.stroke();
    ctx.setLineDash([]);
  });

  // Draw nodes
  filteredSignals.forEach(n=>{
    const r=6+SEV_WEIGHT[n.severity]*3;
    const isSelected=selectedNode===n;
    const isConnected=selectedNode&&filteredEdges.some(e=>(e.source===n.id||e.target===n.id)&&(e.source===selectedNode.id||e.target===selectedNode.id));

    // Glow for selected
    if(isSelected){
      ctx.beginPath();ctx.arc(n.x,n.y,r+8,0,Math.PI*2);
      ctx.fillStyle="rgba(126,184,255,0.3)";ctx.fill();
    }
    if(isConnected){
      ctx.beginPath();ctx.arc(n.x,n.y,r+5,0,Math.PI*2);
      ctx.fillStyle="rgba(126,184,255,0.15)";ctx.fill();
    }

    // Cluster ring
    ctx.beginPath();ctx.arc(n.x,n.y,r+2,0,Math.PI*2);
    ctx.strokeStyle=n.clusterColor;ctx.lineWidth=2;ctx.stroke();

    // Main node
    ctx.beginPath();ctx.arc(n.x,n.y,r,0,Math.PI*2);
    ctx.fillStyle=SEV_COLOR[n.severity];
    if(!isSelected&&!isConnected&&selectedNode)ctx.globalAlpha=0.3;
    ctx.fill();ctx.globalAlpha=1;

    // Source icon
    const icons={drift:"↗",escalation:"⬆",killchain:"⚔",canary:"🐤",covert:"👁",behavior:"🔍"};
    ctx.fillStyle="#000";ctx.font="bold "+(r-2)+"px sans-serif";ctx.textAlign="center";ctx.textBaseline="middle";
    ctx.fillText(icons[n.source]||"?",n.x,n.y);
  });

  ctx.restore();
}

function loop(){simulate();draw();requestAnimationFrame(loop)}

// ── Interaction ──
function screenToWorld(sx,sy){return{x:(sx-panX)/zoom,y:(sy-panY)/zoom}}

canvas.addEventListener("mousedown",e=>{
  const{x,y}=screenToWorld(e.offsetX,e.offsetY);
  // Check node hit
  for(const n of filteredSignals){
    const r=6+SEV_WEIGHT[n.severity]*3;
    if(Math.hypot(n.x-x,n.y-y)<r+4){
      dragNode=n;selectedNode=n;selectedEdge=null;
      showNodeDetails(n);return;
    }
  }
  // Check edge hit
  const nodeMap={};filteredSignals.forEach(n=>{nodeMap[n.id]=n});
  for(const e of filteredEdges){
    const a=nodeMap[e.source],b=nodeMap[e.target];
    if(!a||!b)continue;
    const d=pointToSegDist(x,y,a.x,a.y,b.x,b.y);
    if(d<8){selectedEdge=e;selectedNode=null;showEdgeDetails(e);return}
  }
  isDragging=true;lastMouse={x:e.clientX,y:e.clientY};
  selectedNode=null;selectedEdge=null;
  document.getElementById("sidebar").classList.remove("open");
});

canvas.addEventListener("mousemove",e=>{
  if(dragNode){
    const{x,y}=screenToWorld(e.offsetX,e.offsetY);
    dragNode.x=x;dragNode.y=y;dragNode.vx=0;dragNode.vy=0;
  }else if(isDragging){
    panX+=e.clientX-lastMouse.x;panY+=e.clientY-lastMouse.y;
    lastMouse={x:e.clientX,y:e.clientY};
  }
});

canvas.addEventListener("mouseup",()=>{dragNode=null;isDragging=false});
canvas.addEventListener("wheel",e=>{
  e.preventDefault();
  const scale=e.deltaY>0?0.9:1.1;
  const mx=e.offsetX,my=e.offsetY;
  panX=mx-(mx-panX)*scale;panY=my-(my-panY)*scale;
  zoom*=scale;zoom=Math.max(0.2,Math.min(5,zoom));
},{passive:false});

function pointToSegDist(px,py,ax,ay,bx,by){
  const dx=bx-ax,dy=by-ay,l2=dx*dx+dy*dy;
  if(l2===0)return Math.hypot(px-ax,py-ay);
  let t=((px-ax)*dx+(py-ay)*dy)/l2;t=Math.max(0,Math.min(1,t));
  return Math.hypot(px-(ax+t*dx),py-(ay+t*dy));
}

// ── Sidebar ──
function showNodeDetails(n){
  const sb=document.getElementById("sidebar");
  const sc=document.getElementById("sidebarContent");
  document.getElementById("sidebarTitle").textContent="Signal #"+n.id;
  const sevClass={info:"sev-info",low:"sev-low",medium:"sev-medium",high:"sev-high",critical:"sev-critical"};
  const conns=filteredEdges.filter(e=>e.source===n.id||e.target===n.id);
  sc.innerHTML=`
    <div class="field"><div class="field-label">Source</div><div class="field-value">${n.source}</div></div>
    <div class="field"><div class="field-label">Severity</div><div class="field-value"><span class="severity-badge ${sevClass[n.severity]}">${n.severity}</span></div></div>
    <div class="field"><div class="field-label">Agent</div><div class="field-value">${n.agent}</div></div>
    <div class="field"><div class="field-label">Timestamp</div><div class="field-value">t=${n.timestamp}</div></div>
    <div class="field"><div class="field-label">Description</div><div class="field-value">${n.description}</div></div>
    <div class="field"><div class="field-label">Cluster</div><div class="field-value"><span class="cluster-badge" style="background:${n.clusterColor}">${n.cluster}</span></div></div>
    <div class="field"><div class="field-label">Connections</div><div class="field-value">${conns.length} correlation${conns.length!==1?"s":""}</div></div>
    ${conns.map(e=>{
      const other=e.source===n.id?e.target:e.source;
      const otherNode=data.signals[other];
      return`<div class="field" style="margin-left:12px;border-left:2px solid ${n.clusterColor};padding-left:8px">
        <div class="field-value">→ #${other} (${otherNode.source}, ${otherNode.severity})<br><small>${e.rule} · strength ${e.strength.toFixed(2)}</small></div>
      </div>`;
    }).join("")}
  `;
  sb.classList.add("open");resize();
}

function showEdgeDetails(e){
  const sb=document.getElementById("sidebar");
  const sc=document.getElementById("sidebarContent");
  document.getElementById("sidebarTitle").textContent="Correlation";
  const a=data.signals[e.source],b=data.signals[e.target];
  sc.innerHTML=`
    <div class="field"><div class="field-label">Rule</div><div class="field-value">${e.rule}</div></div>
    <div class="field"><div class="field-label">Strength</div><div class="field-value">${e.strength.toFixed(2)}</div></div>
    <div class="field"><div class="field-label">Same Agent</div><div class="field-value">${e.sameAgent?"Yes":"No"}</div></div>
    <hr style="border-color:#1e2740;margin:12px 0">
    <div class="field"><div class="field-label">Source Signal #${a.id}</div><div class="field-value">${a.source} · ${a.severity} · ${a.agent}<br><small>${a.description}</small></div></div>
    <div class="field"><div class="field-label">Target Signal #${b.id}</div><div class="field-value">${b.source} · ${b.severity} · ${b.agent}<br><small>${b.description}</small></div></div>
    <div class="field"><div class="field-label">Time Delta</div><div class="field-value">${Math.abs(a.timestamp-b.timestamp)} ticks</div></div>
  `;
  sb.classList.add("open");resize();
}

// ── Export ──
document.getElementById("exportBtn").addEventListener("click",()=>{
  const a=document.createElement("a");
  a.download="correlation_graph.png";a.href=canvas.toDataURL("image/png");a.click();
});
document.getElementById("resetBtn").addEventListener("click",()=>{
  panX=0;panY=0;zoom=1;selectedNode=null;selectedEdge=null;
  document.getElementById("sidebar").classList.remove("open");resize();
});

applyFilters();
loop();
})();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
