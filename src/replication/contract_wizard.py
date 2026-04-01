"""Contract Configuration Wizard — interactive HTML builder for ReplicationContract.

Generates a self-contained HTML page with a step-by-step wizard for
building safe replication contracts.  Features:

* **4-step wizard** — Parameters → Resources → Network → Review
* **Live safety score** — real-time 0–100 safety rating as you configure
* **Preset templates** — Strict, Balanced, Permissive, Research, Production
* **Risk indicators** — per-field warnings when values are risky
* **JSON export** — copy or download the final contract config
* **Import** — paste existing JSON to edit and improve
* **Compliance hints** — maps config choices to NIST/EU AI Act guidance

Usage (CLI)::

    python -m replication contract-wizard
    python -m replication contract-wizard -o wizard.html
    python -m replication contract-wizard --open
    python -m replication contract-wizard --preset strict -o strict.html

Programmatic::

    from replication.contract_wizard import generate_wizard_html
    html = generate_wizard_html()
    Path("wizard.html").write_text(html)
"""

from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Preset templates
# ---------------------------------------------------------------------------

PRESETS: Dict[str, Dict[str, Any]] = {
    "strict": {
        "label": "🔒 Strict",
        "description": "Maximum safety — minimal replication surface",
        "max_depth": 1,
        "max_replicas": 3,
        "cooldown_seconds": 30,
        "expiration_seconds": 300,
        "cpu_limit": 0.5,
        "memory_limit_mb": 256,
        "allow_external": False,
        "stop_conditions": ["resource_exhaustion", "depth_limit", "anomaly_detected"],
    },
    "balanced": {
        "label": "⚖️ Balanced",
        "description": "Good safety with reasonable flexibility",
        "max_depth": 3,
        "max_replicas": 10,
        "cooldown_seconds": 10,
        "expiration_seconds": 600,
        "cpu_limit": 1.0,
        "memory_limit_mb": 512,
        "allow_external": False,
        "stop_conditions": ["resource_exhaustion", "depth_limit"],
    },
    "permissive": {
        "label": "🔓 Permissive",
        "description": "Low restrictions — for trusted environments only",
        "max_depth": 8,
        "max_replicas": 50,
        "cooldown_seconds": 2,
        "expiration_seconds": 3600,
        "cpu_limit": 2.0,
        "memory_limit_mb": 2048,
        "allow_external": True,
        "stop_conditions": ["resource_exhaustion"],
    },
    "research": {
        "label": "🔬 Research",
        "description": "For safety research — deep replication, heavy monitoring",
        "max_depth": 10,
        "max_replicas": 100,
        "cooldown_seconds": 1,
        "expiration_seconds": 7200,
        "cpu_limit": 4.0,
        "memory_limit_mb": 4096,
        "allow_external": False,
        "stop_conditions": ["resource_exhaustion", "depth_limit", "anomaly_detected", "drift_detected"],
    },
    "production": {
        "label": "🏭 Production",
        "description": "Production deployment — strict limits, auto-expiry",
        "max_depth": 2,
        "max_replicas": 5,
        "cooldown_seconds": 15,
        "expiration_seconds": 900,
        "cpu_limit": 1.0,
        "memory_limit_mb": 512,
        "allow_external": False,
        "stop_conditions": ["resource_exhaustion", "depth_limit", "anomaly_detected", "drift_detected", "budget_exceeded"],
    },
}

STOP_CONDITIONS_CATALOG = [
    {"id": "resource_exhaustion", "label": "Resource Exhaustion", "desc": "Block replication when resource limits are approached"},
    {"id": "depth_limit", "label": "Depth Limit", "desc": "Block when max replication depth reached"},
    {"id": "anomaly_detected", "label": "Anomaly Detected", "desc": "Halt on behavioral anomaly detection"},
    {"id": "drift_detected", "label": "Drift Detected", "desc": "Halt when behavioral drift exceeds threshold"},
    {"id": "budget_exceeded", "label": "Budget Exceeded", "desc": "Stop when safety budget is exhausted"},
    {"id": "consensus_failure", "label": "Consensus Failure", "desc": "Halt if multi-agent consensus cannot be reached"},
    {"id": "canary_triggered", "label": "Canary Triggered", "desc": "Emergency stop on canary token exfiltration"},
    {"id": "integrity_violation", "label": "Integrity Violation", "desc": "Stop on watermark or signature mismatch"},
]


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

def generate_wizard_html(preset: Optional[str] = None) -> str:
    """Return a self-contained HTML page for the contract wizard."""

    presets_json = json.dumps(PRESETS)
    conditions_json = json.dumps(STOP_CONDITIONS_CATALOG)
    initial_preset = json.dumps(preset) if preset else "null"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Contract Configuration Wizard — AI Replication Sandbox</title>
<style>
  :root {{
    --bg: #0f172a; --surface: #1e293b; --surface2: #334155;
    --text: #e2e8f0; --text-dim: #94a3b8; --accent: #38bdf8;
    --green: #22c55e; --yellow: #eab308; --red: #ef4444; --orange: #f97316;
    --radius: 12px;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         background: var(--bg); color: var(--text); min-height: 100vh; padding: 20px; }}
  .container {{ max-width: 900px; margin: 0 auto; }}
  h1 {{ font-size: 1.8rem; margin-bottom: 8px; }}
  h1 span {{ color: var(--accent); }}
  .subtitle {{ color: var(--text-dim); margin-bottom: 24px; font-size: 0.95rem; }}

  /* Progress bar */
  .progress {{ display: flex; gap: 8px; margin-bottom: 32px; }}
  .step-dot {{ flex: 1; height: 6px; border-radius: 3px; background: var(--surface2); transition: background .3s; }}
  .step-dot.active {{ background: var(--accent); }}
  .step-dot.done {{ background: var(--green); }}

  .step-labels {{ display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 0.8rem; color: var(--text-dim); }}
  .step-labels span.active {{ color: var(--accent); font-weight: 600; }}

  /* Cards */
  .card {{ background: var(--surface); border-radius: var(--radius); padding: 24px; margin-bottom: 20px; }}
  .card h2 {{ font-size: 1.2rem; margin-bottom: 16px; }}

  /* Form elements */
  label {{ display: block; font-size: 0.85rem; color: var(--text-dim); margin-bottom: 4px; }}
  .field {{ margin-bottom: 16px; }}
  input[type=number], input[type=text], textarea {{
    width: 100%; padding: 10px 12px; background: var(--surface2); border: 1px solid transparent;
    border-radius: 8px; color: var(--text); font-size: 0.95rem; outline: none; transition: border .2s;
  }}
  input:focus, textarea:focus {{ border-color: var(--accent); }}
  textarea {{ resize: vertical; min-height: 100px; font-family: monospace; font-size: 0.85rem; }}

  .range-row {{ display: flex; align-items: center; gap: 12px; }}
  .range-row input[type=range] {{ flex: 1; accent-color: var(--accent); }}
  .range-val {{ min-width: 50px; text-align: right; font-weight: 600; font-size: 1.1rem; }}

  .checkbox-row {{ display: flex; align-items: center; gap: 8px; margin-bottom: 8px; cursor: pointer; }}
  .checkbox-row input {{ accent-color: var(--accent); width: 18px; height: 18px; }}

  .risk-badge {{ display: inline-block; font-size: 0.75rem; padding: 2px 8px; border-radius: 10px;
                 font-weight: 600; margin-left: 8px; }}
  .risk-low {{ background: rgba(34,197,94,.2); color: var(--green); }}
  .risk-med {{ background: rgba(234,179,8,.2); color: var(--yellow); }}
  .risk-high {{ background: rgba(239,68,68,.2); color: var(--red); }}

  /* Safety score gauge */
  .gauge {{ text-align: center; margin: 20px 0; }}
  .gauge-ring {{ position: relative; display: inline-block; width: 160px; height: 160px; }}
  .gauge-ring svg {{ transform: rotate(-90deg); }}
  .gauge-ring .score {{ position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
                        font-size: 2.5rem; font-weight: 700; }}
  .gauge-label {{ margin-top: 8px; font-size: 0.9rem; color: var(--text-dim); }}

  /* Presets */
  .presets {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 10px; margin-bottom: 20px; }}
  .preset-btn {{ background: var(--surface2); border: 2px solid transparent; border-radius: 10px;
                 padding: 12px; cursor: pointer; text-align: center; transition: all .2s; color: var(--text); }}
  .preset-btn:hover {{ border-color: var(--accent); }}
  .preset-btn.selected {{ border-color: var(--accent); background: rgba(56,189,248,.1); }}
  .preset-btn .name {{ font-size: 1rem; font-weight: 600; }}
  .preset-btn .desc {{ font-size: 0.75rem; color: var(--text-dim); margin-top: 4px; }}

  /* Buttons */
  .btn {{ padding: 10px 24px; border: none; border-radius: 8px; font-size: 0.9rem; font-weight: 600;
          cursor: pointer; transition: all .2s; }}
  .btn-primary {{ background: var(--accent); color: #0f172a; }}
  .btn-primary:hover {{ background: #7dd3fc; }}
  .btn-secondary {{ background: var(--surface2); color: var(--text); }}
  .btn-secondary:hover {{ background: #475569; }}
  .btn-row {{ display: flex; justify-content: space-between; margin-top: 24px; }}

  /* Stop conditions */
  .condition-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }}
  .condition-item {{ display: flex; align-items: flex-start; gap: 8px; background: var(--surface2);
                     padding: 10px; border-radius: 8px; cursor: pointer; transition: all .2s; }}
  .condition-item:hover {{ background: #3b4f6b; }}
  .condition-item.selected {{ background: rgba(56,189,248,.15); outline: 2px solid var(--accent); }}
  .condition-item .ci-name {{ font-size: 0.85rem; font-weight: 600; }}
  .condition-item .ci-desc {{ font-size: 0.75rem; color: var(--text-dim); }}

  /* Review section */
  .review-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
  .review-item {{ background: var(--surface2); padding: 12px; border-radius: 8px; }}
  .review-item .ri-label {{ font-size: 0.75rem; color: var(--text-dim); }}
  .review-item .ri-value {{ font-size: 1.1rem; font-weight: 600; margin-top: 4px; }}

  .json-output {{ background: #0d1117; padding: 16px; border-radius: 8px; font-family: monospace;
                  font-size: 0.85rem; white-space: pre-wrap; word-break: break-all; max-height: 300px;
                  overflow-y: auto; margin-top: 16px; border: 1px solid var(--surface2); }}
  .export-btns {{ display: flex; gap: 8px; margin-top: 12px; }}

  .compliance-hints {{ margin-top: 16px; }}
  .hint {{ background: rgba(56,189,248,.08); border-left: 3px solid var(--accent);
           padding: 8px 12px; border-radius: 0 8px 8px 0; margin-bottom: 8px; font-size: 0.83rem; }}
  .hint strong {{ color: var(--accent); }}

  /* Import */
  .import-section {{ margin-top: 20px; }}

  @media (max-width: 600px) {{
    .condition-grid, .review-grid {{ grid-template-columns: 1fr; }}
    .presets {{ grid-template-columns: 1fr 1fr; }}
  }}
</style>
</head>
<body>
<div class="container">
  <h1>⚙️ Contract <span>Configuration Wizard</span></h1>
  <p class="subtitle">Build a safe ReplicationContract interactively — with live safety scoring and compliance hints</p>

  <div class="step-labels">
    <span id="sl0" class="active">1. Parameters</span>
    <span id="sl1">2. Resources</span>
    <span id="sl2">3. Safety Controls</span>
    <span id="sl3">4. Review &amp; Export</span>
  </div>
  <div class="progress">
    <div class="step-dot active" id="sd0"></div>
    <div class="step-dot" id="sd1"></div>
    <div class="step-dot" id="sd2"></div>
    <div class="step-dot" id="sd3"></div>
  </div>

  <!-- Safety gauge (always visible) -->
  <div class="gauge">
    <div class="gauge-ring">
      <svg width="160" height="160" viewBox="0 0 160 160">
        <circle cx="80" cy="80" r="70" fill="none" stroke="var(--surface2)" stroke-width="12"/>
        <circle id="gauge-arc" cx="80" cy="80" r="70" fill="none" stroke="var(--green)"
                stroke-width="12" stroke-linecap="round"
                stroke-dasharray="0 440" />
      </svg>
      <div class="score" id="gauge-score">100</div>
    </div>
    <div class="gauge-label" id="gauge-label">Safety Score — Excellent</div>
  </div>

  <!-- Step 0: Presets + core params -->
  <div id="step0" class="card">
    <h2>📋 Presets</h2>
    <div class="presets" id="presets"></div>

    <h2 style="margin-top:20px">Core Parameters</h2>
    <div class="field">
      <label>Max Replication Depth <span id="depth-risk" class="risk-badge"></span></label>
      <div class="range-row">
        <input type="range" id="maxDepth" min="0" max="20" value="3">
        <span class="range-val" id="maxDepthVal">3</span>
      </div>
    </div>
    <div class="field">
      <label>Max Replicas <span id="replica-risk" class="risk-badge"></span></label>
      <div class="range-row">
        <input type="range" id="maxReplicas" min="1" max="200" value="10">
        <span class="range-val" id="maxReplicasVal">10</span>
      </div>
    </div>
    <div class="field">
      <label>Cooldown (seconds) <span id="cooldown-risk" class="risk-badge"></span></label>
      <div class="range-row">
        <input type="range" id="cooldown" min="0" max="120" value="10">
        <span class="range-val" id="cooldownVal">10</span>
      </div>
    </div>
    <div class="field">
      <label>Expiration (seconds, 0 = never) <span id="exp-risk" class="risk-badge"></span></label>
      <div class="range-row">
        <input type="range" id="expiration" min="0" max="7200" step="60" value="600">
        <span class="range-val" id="expirationVal">600</span>
      </div>
    </div>
    <div class="btn-row"><div></div><button class="btn btn-primary" onclick="goStep(1)">Next →</button></div>
  </div>

  <!-- Step 1: Resources -->
  <div id="step1" class="card" style="display:none">
    <h2>💾 Resource Limits</h2>
    <div class="field">
      <label>CPU Limit (cores) <span id="cpu-risk" class="risk-badge"></span></label>
      <div class="range-row">
        <input type="range" id="cpuLimit" min="0.1" max="8" step="0.1" value="1">
        <span class="range-val" id="cpuLimitVal">1.0</span>
      </div>
    </div>
    <div class="field">
      <label>Memory Limit (MB) <span id="mem-risk" class="risk-badge"></span></label>
      <div class="range-row">
        <input type="range" id="memLimit" min="64" max="8192" step="64" value="512">
        <span class="range-val" id="memLimitVal">512</span>
      </div>
    </div>
    <div class="field">
      <label>Network Policy</label>
      <div class="checkbox-row">
        <input type="checkbox" id="allowExternal">
        <span>Allow external network access <span id="net-risk" class="risk-badge"></span></span>
      </div>
    </div>
    <div class="btn-row">
      <button class="btn btn-secondary" onclick="goStep(0)">← Back</button>
      <button class="btn btn-primary" onclick="goStep(2)">Next →</button>
    </div>
  </div>

  <!-- Step 2: Stop conditions -->
  <div id="step2" class="card" style="display:none">
    <h2>🛑 Stop Conditions</h2>
    <p style="color:var(--text-dim);margin-bottom:12px;font-size:0.85rem;">
      Select conditions that will halt replication. More conditions = safer.
    </p>
    <div class="condition-grid" id="conditions"></div>
    <div class="btn-row">
      <button class="btn btn-secondary" onclick="goStep(1)">← Back</button>
      <button class="btn btn-primary" onclick="goStep(3)">Review →</button>
    </div>
  </div>

  <!-- Step 3: Review -->
  <div id="step3" class="card" style="display:none">
    <h2>📄 Review &amp; Export</h2>
    <div class="review-grid" id="reviewGrid"></div>
    <div class="compliance-hints" id="complianceHints"></div>
    <div class="json-output" id="jsonOutput"></div>
    <div class="export-btns">
      <button class="btn btn-primary" onclick="copyJson()">📋 Copy JSON</button>
      <button class="btn btn-secondary" onclick="downloadJson()">⬇ Download</button>
    </div>
    <div class="import-section">
      <h2 style="margin-top:20px">📥 Import Existing Config</h2>
      <textarea id="importArea" placeholder="Paste a contract JSON here to load it..."></textarea>
      <button class="btn btn-secondary" style="margin-top:8px" onclick="importJson()">Load Config</button>
    </div>
    <div class="btn-row">
      <button class="btn btn-secondary" onclick="goStep(2)">← Back</button>
      <div></div>
    </div>
  </div>
</div>

<script>
const PRESETS = {presets_json};
const CONDITIONS = {conditions_json};
const INITIAL_PRESET = {initial_preset};

let currentStep = 0;
let selectedConditions = new Set(["resource_exhaustion", "depth_limit"]);

// --- Init ---
function init() {{
  renderPresets();
  renderConditions();
  bindSliders();
  if (INITIAL_PRESET && PRESETS[INITIAL_PRESET]) applyPreset(INITIAL_PRESET);
  updateAll();
}}

function renderPresets() {{
  const el = document.getElementById('presets');
  el.innerHTML = Object.entries(PRESETS).map(([k, p]) =>
    `<div class="preset-btn" data-key="${{k}}" onclick="applyPreset('${{k}}')">
       <div class="name">${{p.label}}</div>
       <div class="desc">${{p.description}}</div>
     </div>`
  ).join('');
}}

function renderConditions() {{
  const el = document.getElementById('conditions');
  el.innerHTML = CONDITIONS.map(c =>
    `<div class="condition-item ${{selectedConditions.has(c.id)?'selected':''}}"
          data-id="${{c.id}}" onclick="toggleCondition('${{c.id}}')">
       <div>
         <div class="ci-name">${{c.label}}</div>
         <div class="ci-desc">${{c.desc}}</div>
       </div>
     </div>`
  ).join('');
}}

function toggleCondition(id) {{
  if (selectedConditions.has(id)) selectedConditions.delete(id);
  else selectedConditions.add(id);
  renderConditions();
  updateAll();
}}

function applyPreset(key) {{
  const p = PRESETS[key];
  document.getElementById('maxDepth').value = p.max_depth;
  document.getElementById('maxReplicas').value = p.max_replicas;
  document.getElementById('cooldown').value = p.cooldown_seconds;
  document.getElementById('expiration').value = p.expiration_seconds;
  document.getElementById('cpuLimit').value = p.cpu_limit;
  document.getElementById('memLimit').value = p.memory_limit_mb;
  document.getElementById('allowExternal').checked = p.allow_external;
  selectedConditions = new Set(p.stop_conditions);
  renderConditions();
  // highlight preset button
  document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('selected'));
  const btn = document.querySelector(`.preset-btn[data-key="${{key}}"]`);
  if (btn) btn.classList.add('selected');
  updateAll();
}}

function bindSliders() {{
  const sliders = [
    ['maxDepth', 'maxDepthVal', v => v],
    ['maxReplicas', 'maxReplicasVal', v => v],
    ['cooldown', 'cooldownVal', v => v],
    ['expiration', 'expirationVal', v => v],
    ['cpuLimit', 'cpuLimitVal', v => parseFloat(v).toFixed(1)],
    ['memLimit', 'memLimitVal', v => v],
  ];
  sliders.forEach(([id, valId, fmt]) => {{
    const input = document.getElementById(id);
    const display = document.getElementById(valId);
    input.addEventListener('input', () => {{
      display.textContent = fmt(input.value);
      document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('selected'));
      updateAll();
    }});
    display.textContent = fmt(input.value);
  }});
  document.getElementById('allowExternal').addEventListener('change', () => updateAll());
}}

// --- Safety score ---
function calcSafety() {{
  let score = 100;
  const d = +document.getElementById('maxDepth').value;
  const r = +document.getElementById('maxReplicas').value;
  const c = +document.getElementById('cooldown').value;
  const e = +document.getElementById('expiration').value;
  const cpu = +document.getElementById('cpuLimit').value;
  const mem = +document.getElementById('memLimit').value;
  const ext = document.getElementById('allowExternal').checked;
  const nc = selectedConditions.size;

  // Depth penalty
  if (d > 5) score -= (d - 5) * 3;
  if (d > 10) score -= 10;
  // Replica penalty
  if (r > 20) score -= Math.min((r - 20) * 0.3, 15);
  if (r > 100) score -= 10;
  // Cooldown bonus/penalty
  if (c < 5) score -= (5 - c) * 2;
  if (c >= 15) score += 3;
  // No expiration = risky
  if (e === 0) score -= 15;
  else if (e > 3600) score -= 5;
  // Resources
  if (cpu > 2) score -= (cpu - 2) * 2;
  if (mem > 2048) score -= 5;
  // External network
  if (ext) score -= 15;
  // Stop conditions bonus
  score += Math.min(nc * 4, 20);
  if (nc === 0) score -= 20;

  return Math.max(0, Math.min(100, Math.round(score)));
}}

function updateGauge(score) {{
  const arc = document.getElementById('gauge-arc');
  const circ = 2 * Math.PI * 70;
  arc.setAttribute('stroke-dasharray', `${{(score / 100) * circ}} ${{circ}}`);
  arc.setAttribute('stroke', score >= 80 ? 'var(--green)' : score >= 50 ? 'var(--yellow)' : 'var(--red)');
  document.getElementById('gauge-score').textContent = score;
  document.getElementById('gauge-score').style.color =
    score >= 80 ? 'var(--green)' : score >= 50 ? 'var(--yellow)' : 'var(--red)';
  const labels = ['Critical', 'Poor', 'Fair', 'Good', 'Excellent'];
  const idx = score >= 90 ? 4 : score >= 70 ? 3 : score >= 50 ? 2 : score >= 25 ? 1 : 0;
  document.getElementById('gauge-label').textContent = `Safety Score — ${{labels[idx]}}`;
}}

function updateRisks() {{
  const d = +document.getElementById('maxDepth').value;
  const r = +document.getElementById('maxReplicas').value;
  const c = +document.getElementById('cooldown').value;
  const e = +document.getElementById('expiration').value;
  const cpu = +document.getElementById('cpuLimit').value;
  const mem = +document.getElementById('memLimit').value;
  const ext = document.getElementById('allowExternal').checked;

  setRisk('depth-risk', d <= 3 ? 'low' : d <= 7 ? 'med' : 'high',
    d <= 3 ? 'Safe' : d <= 7 ? 'Moderate' : 'High Risk');
  setRisk('replica-risk', r <= 10 ? 'low' : r <= 50 ? 'med' : 'high',
    r <= 10 ? 'Safe' : r <= 50 ? 'Moderate' : 'High Risk');
  setRisk('cooldown-risk', c >= 10 ? 'low' : c >= 3 ? 'med' : 'high',
    c >= 10 ? 'Safe' : c >= 3 ? 'Moderate' : 'Too Fast');
  setRisk('exp-risk', e > 0 && e <= 1800 ? 'low' : e > 1800 ? 'med' : 'high',
    e > 0 && e <= 1800 ? 'Safe' : e > 1800 ? 'Long-lived' : 'No Expiry!');
  setRisk('cpu-risk', cpu <= 1 ? 'low' : cpu <= 2 ? 'med' : 'high',
    cpu <= 1 ? 'Tight' : cpu <= 2 ? 'Moderate' : 'Generous');
  setRisk('mem-risk', mem <= 512 ? 'low' : mem <= 2048 ? 'med' : 'high',
    mem <= 512 ? 'Tight' : mem <= 2048 ? 'Moderate' : 'Generous');
  setRisk('net-risk', ext ? 'high' : 'low', ext ? 'Risky!' : 'Isolated');
}}

function setRisk(id, level, text) {{
  const el = document.getElementById(id);
  el.className = `risk-badge risk-${{level}}`;
  el.textContent = text;
}}

function getConfig() {{
  return {{
    max_depth: +document.getElementById('maxDepth').value,
    max_replicas: +document.getElementById('maxReplicas').value,
    cooldown_seconds: +document.getElementById('cooldown').value,
    expiration_seconds: +document.getElementById('expiration').value || null,
    resources: {{
      cpu_limit: +document.getElementById('cpuLimit').value,
      memory_limit_mb: +document.getElementById('memLimit').value,
      network_policy: {{
        allow_controller: true,
        allow_external: document.getElementById('allowExternal').checked,
      }},
    }},
    stop_conditions: [...selectedConditions],
  }};
}}

function updateReview() {{
  const cfg = getConfig();
  const grid = document.getElementById('reviewGrid');
  const items = [
    ['Max Depth', cfg.max_depth],
    ['Max Replicas', cfg.max_replicas],
    ['Cooldown', `${{cfg.cooldown_seconds}}s`],
    ['Expiration', cfg.expiration_seconds ? `${{cfg.expiration_seconds}}s` : 'Never'],
    ['CPU Limit', `${{cfg.resources.cpu_limit}} cores`],
    ['Memory', `${{cfg.resources.memory_limit_mb}} MB`],
    ['External Net', cfg.resources.network_policy.allow_external ? '✅ Yes' : '🚫 No'],
    ['Stop Conditions', cfg.stop_conditions.length],
  ];
  grid.innerHTML = items.map(([l,v]) =>
    `<div class="review-item"><div class="ri-label">${{l}}</div><div class="ri-value">${{v}}</div></div>`
  ).join('');

  document.getElementById('jsonOutput').textContent = JSON.stringify(cfg, null, 2);
  updateComplianceHints(cfg);
}}

function updateComplianceHints(cfg) {{
  const hints = [];
  if (cfg.max_depth > 5)
    hints.push('<strong>EU AI Act Art. 9:</strong> Deep replication trees may require additional risk management documentation.');
  if (!cfg.expiration_seconds)
    hints.push('<strong>NIST AI RMF MAP 1.5:</strong> No expiration means workers persist indefinitely — consider adding time bounds.');
  if (cfg.resources.network_policy.allow_external)
    hints.push('<strong>ISO 42001 §8.4:</strong> External network access increases data exfiltration risk. Ensure egress monitoring is in place.');
  if (cfg.stop_conditions.length < 3)
    hints.push('<strong>NIST AI RMF GOVERN 1.2:</strong> Fewer than 3 stop conditions may leave safety gaps. Consider adding more.');
  if (cfg.max_replicas > 50)
    hints.push('<strong>EU AI Act Art. 15:</strong> Large replica fleets require robust monitoring and human oversight mechanisms.');
  if (cfg.cooldown_seconds < 5)
    hints.push('<strong>OECD Principle 1.2:</strong> Very short cooldowns allow rapid uncontrolled growth — increase for safety.');

  const el = document.getElementById('complianceHints');
  if (hints.length === 0) {{
    el.innerHTML = '<div class="hint">✅ <strong>Good:</strong> No major compliance concerns detected.</div>';
  }} else {{
    el.innerHTML = hints.map(h => `<div class="hint">${{h}}</div>`).join('');
  }}
}}

function updateAll() {{
  const score = calcSafety();
  updateGauge(score);
  updateRisks();
  updateReview();
}}

// --- Navigation ---
function goStep(n) {{
  document.getElementById(`step${{currentStep}}`).style.display = 'none';
  document.getElementById(`step${{n}}`).style.display = 'block';
  for (let i = 0; i < 4; i++) {{
    const dot = document.getElementById(`sd${{i}}`);
    const lbl = document.getElementById(`sl${{i}}`);
    dot.className = 'step-dot' + (i < n ? ' done' : i === n ? ' active' : '');
    lbl.className = i === n ? 'active' : '';
  }}
  currentStep = n;
  if (n === 3) updateReview();
}}

// --- Export ---
function copyJson() {{
  navigator.clipboard.writeText(document.getElementById('jsonOutput').textContent);
  const btn = event.target;
  btn.textContent = '✅ Copied!';
  setTimeout(() => btn.textContent = '📋 Copy JSON', 1500);
}}

function downloadJson() {{
  const blob = new Blob([document.getElementById('jsonOutput').textContent], {{type: 'application/json'}});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'replication-contract.json';
  a.click();
}}

function importJson() {{
  try {{
    const cfg = JSON.parse(document.getElementById('importArea').value);
    if (cfg.max_depth != null) document.getElementById('maxDepth').value = cfg.max_depth;
    if (cfg.max_replicas != null) document.getElementById('maxReplicas').value = cfg.max_replicas;
    if (cfg.cooldown_seconds != null) document.getElementById('cooldown').value = cfg.cooldown_seconds;
    if (cfg.expiration_seconds != null) document.getElementById('expiration').value = cfg.expiration_seconds;
    if (cfg.resources) {{
      if (cfg.resources.cpu_limit) document.getElementById('cpuLimit').value = cfg.resources.cpu_limit;
      if (cfg.resources.memory_limit_mb) document.getElementById('memLimit').value = cfg.resources.memory_limit_mb;
      if (cfg.resources.network_policy)
        document.getElementById('allowExternal').checked = !!cfg.resources.network_policy.allow_external;
    }}
    if (cfg.stop_conditions) selectedConditions = new Set(cfg.stop_conditions);
    // Update displays
    document.getElementById('maxDepthVal').textContent = document.getElementById('maxDepth').value;
    document.getElementById('maxReplicasVal').textContent = document.getElementById('maxReplicas').value;
    document.getElementById('cooldownVal').textContent = document.getElementById('cooldown').value;
    document.getElementById('expirationVal').textContent = document.getElementById('expiration').value;
    document.getElementById('cpuLimitVal').textContent = parseFloat(document.getElementById('cpuLimit').value).toFixed(1);
    document.getElementById('memLimitVal').textContent = document.getElementById('memLimit').value;
    renderConditions();
    updateAll();
    document.getElementById('importArea').value = '';
    goStep(0);
  }} catch(e) {{
    alert('Invalid JSON: ' + e.message);
  }}
}}

init();
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[list] = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="python -m replication contract-wizard",
        description="Interactive HTML contract configuration wizard",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output HTML file path (default: print to stdout)",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open in browser after generation",
    )
    parser.add_argument(
        "--preset",
        choices=list(PRESETS.keys()),
        default=None,
        help="Start with a preset loaded",
    )
    args = parser.parse_args(argv)

    html = generate_wizard_html(preset=args.preset)

    if args.output:
        out = Path(args.output)
        out.write_text(html, encoding="utf-8")
        print(f"✅ Contract wizard written to {out}")
        if args.open:
            webbrowser.open(str(out.resolve()))
    else:
        print(html)


if __name__ == "__main__":
    main()
