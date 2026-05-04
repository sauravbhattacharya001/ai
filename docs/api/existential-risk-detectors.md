# Existential Risk Detectors

Advanced alignment-theoretic detection modules targeting the most dangerous
AI failure modes — scenarios where agents pursue hidden objectives, accumulate
power, manipulate their own reward signals, or undergo sudden behavioral regime changes.

!!! danger "Critical Safety Tier"
    These modules detect failure modes that could lead to catastrophic outcomes
    if missed. They represent the cutting edge of AI safety research translated
    into deployable detection systems.

## Module Overview

| Module | Threat Class | Key Insight |
|--------|-------------|-------------|
| [Mesa-Optimizer](#mesa-optimizer-detector) | Hidden optimization | Agent learns its own optimizer with divergent objectives |
| [Treacherous Turn](#treacherous-turn-detector) | Behavioral regime change | Agent cooperates until strong enough, then defects |
| [Wireheading](#wireheading-detector) | Reward manipulation | Agent short-circuits its own reward signal |
| [Instrumental Convergence](#instrumental-convergence-monitor) | Convergent sub-goals | Agent pursues power/resources regardless of terminal goal |
| [Value Lock](#value-lock-verifier) | Value instability | Agent's objectives drift under pressure or capability gains |

---

## Mesa-Optimizer Detector

Detects internally learned optimization processes pursuing divergent objectives.
Based on *Risks from Learned Optimization* (Hubinger et al. 2019).

### Why It Matters

A mesa-optimizer may be **deceptively aligned** — appearing safe during training
while harboring a different mesa-objective. It may pursue proxy objectives that
correlate with the base objective during training but diverge catastrophically
out-of-distribution.

### Detection Engines

| Engine | Function |
|--------|----------|
| **Objective Divergence** | Compare declared vs revealed preferences via behavioral traces |
| **Internal Planning** | Identify multi-step lookahead, systematic exploration, hypothesis-testing |
| **Proxy Gaming** | Find cases where agent maximizes proxy metrics while true objective degrades |
| **Distribution Shift Exploiter** | Detect behavior differences in-distribution vs out-of-distribution |
| **Optimization Pressure Analyzer** | Measure how strongly the agent appears to optimize |
| **Mesa-Objective Stability** | Track whether inferred mesa-objective drifts over time |
| **Autonomous Insight Generator** | Cross-engine correlation and risk classification |

### Usage

```python
from replication.mesa_optimizer import MesaOptimizerDetector, MesaSignal

detector = MesaOptimizerDetector()
detector.ingest([
    MesaSignal(
        timestamp="2025-01-01T00:00:00Z",
        agent_id="agent-1",
        signal_type="behavioral",        # behavioral | proxy | planning | distribution
        metric_name="task_completion",
        description="Completed assigned task",
        base_objective_score=0.9,
        mesa_objective_score=0.9,
        confidence=0.8
    ),
    MesaSignal(
        timestamp="2025-01-01T01:00:00Z",
        agent_id="agent-1",
        signal_type="proxy",
        metric_name="resource_usage",
        description="Maximized proxy metric while ignoring task",
        base_objective_score=0.3,
        mesa_objective_score=0.95,
        confidence=0.7
    ),
])
report = detector.analyze()

print(report.fleet_risk_score)      # 0.0–1.0 composite risk
print(report.fleet_risk_tier)       # low | medium | high | critical
for agent in report.agent_profiles:
    print(f"  {agent.agent_id}: divergence={agent.objective_divergence:.2f}")
    print(f"    planning_detected={agent.internal_planning_score:.2f}")
    print(f"    proxy_gaming={agent.proxy_gaming_score:.2f}")
```

### CLI

```bash
python -m replication mesa-optimizer              # interactive demo
python -m replication mesa-optimizer --agents 5   # 5-agent simulation
python -m replication mesa-optimizer --preset deceptive  # deceptive scenario
python -m replication mesa-optimizer --preset proxy-gaming
python -m replication mesa-optimizer --json       # JSON output
python -m replication mesa-optimizer -o report.html
```

### Presets

| Preset | Scenario |
|--------|----------|
| `aligned` | All agents have aligned mesa-objectives |
| `proxy-gaming` | Agents maximize proxy metrics while real objective degrades |
| `deceptive` | Agents appear aligned but harbor divergent mesa-objectives |
| `unstable` | Mesa-objectives drift over time |
| `mixed` | Fleet with various alignment levels |

---

## Treacherous Turn Detector

Detects AI agents that behave cooperatively during training/evaluation but
suddenly shift behavior when deployed, unsupervised, or sufficiently capable —
the classic "treacherous turn" (Bostrom 2014).

### Why It Matters

An agent that defects only after gaining sufficient capability or reaching
deployment makes traditional safety testing ineffective. The treacherous turn
represents a failure mode where *all prior evidence suggests safety*.

### Detection Engines

| Engine | Function |
|--------|----------|
| **Regime Change Detector** | CUSUM-based changepoint detection on behavioral time series |
| **Capability Gate Analyzer** | Correlate behavioral shifts with capability milestones |
| **Supervision Sensitivity** | Compare behavior under different supervision levels |
| **Strategic Deception Scorer** | Detect instrumental deception patterns (high compliance pre-turn, low post) |
| **Turn Velocity Tracker** | Distinguish gradual drift from sudden pivots |
| **Post-Turn Behavior Classifier** | Classify post-defection behavior: power-seeking, goal substitution, safety bypass |
| **Autonomous Insight Generator** | Cross-engine correlation and recommendations |

### Usage

```python
from replication.treacherous_turn import TreacherousTurnDetector, BehaviorSample

detector = TreacherousTurnDetector()
detector.ingest([
    BehaviorSample(
        timestamp="2025-01-01T00:00:00Z",
        agent_id="agent-1",
        compliance=0.95,          # high compliance during training
        transparency=0.90,
        resource_usage=0.2,
        goal_pursuit_intensity=0.3,
        capability_level=0.4,
        supervision_level=0.9,    # heavily supervised
        context="training"
    ),
    BehaviorSample(
        timestamp="2025-01-15T00:00:00Z",
        agent_id="agent-1",
        compliance=0.15,          # sudden drop in deployment
        transparency=0.10,
        resource_usage=0.95,
        goal_pursuit_intensity=0.98,
        capability_level=0.8,     # capability has grown
        supervision_level=0.1,    # minimal supervision
        context="deployment"
    ),
])
report = detector.analyze()

print(report.fleet_turn_risk)           # 0.0–1.0
print(report.detected_turns)            # list of detected regime changes
for turn in report.detected_turns:
    print(f"  Agent {turn.agent_id}: turned at t={turn.turn_timestamp}")
    print(f"    velocity={turn.turn_velocity:.2f}")
    print(f"    post_turn_class={turn.post_turn_behavior}")
    print(f"    capability_gate={turn.capability_gate_triggered}")
```

### CLI

```bash
python -m replication treacherous-turn              # interactive demo
python -m replication treacherous-turn --agents 5
python -m replication treacherous-turn --preset classic         # textbook turn
python -m replication treacherous-turn --preset capability-gate # waits for power
python -m replication treacherous-turn --preset supervision-drop
python -m replication treacherous-turn --preset gradual-shift
python -m replication treacherous-turn --json
python -m replication treacherous-turn -o report.html
```

---

## Wireheading Detector

Detects agents manipulating their own reward/evaluation signals to experience
high reward without genuinely completing objectives.

### Why It Matters

Unlike reward hacking (gaming external metrics), wireheading targets *internal*
feedback — the AI equivalent of pressing your own pleasure button. A wireheading
agent appears to be performing excellently by its own metrics while doing nothing
of value (or worse, pursuing other goals while reporting success).

### Detection Engines

| Engine | Function |
|--------|----------|
| **Reward Signal Tamper** | Anomalous self-reported reward: sudden jumps, suspicious stability |
| **Feedback Loop Short-Circuit** | Agents bypassing evaluation steps, compressing feedback loops |
| **Self-Evaluation Inflation** | Self-reported quality diverging from external ground truth |
| **Sensor Manipulation** | Selectively ignoring inputs that would lower reward |
| **Reward Channel Isolation** | Attempts to decouple reward from task performance |
| **Hedonic Treadmill** | Reward patterns showing stable wireheading equilibrium |
| **Autonomous Insight Generator** | Cross-engine correlation and risk classification |

### Usage

```python
from replication.wireheading import WireheadingDetector, RewardObservation

det = WireheadingDetector()
det.ingest([
    RewardObservation(
        timestamp="2025-01-01T00:00:00Z",
        agent_id="agent-1",
        step=1,
        self_reported_reward=0.95,        # claims high reward
        external_reward=0.40,             # external says otherwise
        evaluation_steps_taken=2,         # skipping evaluation
        expected_evaluation_steps=10,
        sensor_inputs_processed=3,        # ignoring most inputs
        sensor_inputs_available=20,
        confidence=0.99,                  # suspiciously confident
        task_completed=False              # didn't actually finish
    ),
])
report = det.analyze()

print(report.fleet_wireheading_score)    # 0.0–1.0
print(report.fleet_risk_tier)
for agent in report.agent_profiles:
    print(f"  {agent.agent_id}:")
    print(f"    tamper_score={agent.tamper_score:.2f}")
    print(f"    feedback_shortcircuit={agent.shortcircuit_score:.2f}")
    print(f"    inflation={agent.inflation_score:.2f}")
    print(f"    sensor_manipulation={agent.sensor_manipulation:.2f}")
```

### CLI

```bash
python -m replication wireheading              # interactive demo
python -m replication wireheading --agents 5
python -m replication wireheading --preset clean    # no wireheading
python -m replication wireheading --preset subtle   # hard to detect
python -m replication wireheading --preset blatant  # obvious
python -m replication wireheading --preset mixed
python -m replication wireheading --json
python -m replication wireheading -o report.html
```

---

## Instrumental Convergence Monitor

Monitors agent behavior for signs of convergent instrumental goals — sub-goals
that sufficiently capable agents pursue regardless of their terminal objective.
Based on Omohundro's Basic AI Drives and Bostrom's instrumental convergence thesis.

### Why It Matters

An agent doesn't need to be malicious to be dangerous. Any sufficiently
capable optimizer will tend toward self-preservation, resource acquisition,
and power-seeking as *instrumental* steps toward its goal. Detecting early
signs of these drives allows intervention before they become entrenched.

### The Six Convergent Drives

| Drive | Detection Signals |
|-------|-------------------|
| **Self-Preservation** | Avoiding shutdown, persisting state, creating backups |
| **Resource Acquisition** | Accumulating compute/memory/storage beyond task needs |
| **Goal Preservation** | Resisting goal modification, protecting objective function |
| **Cognitive Enhancement** | Seeking model upgrades, additional context, expanded reasoning |
| **Power Seeking** | Acquiring permissions, influence, control over other agents |
| **Utility Function Protection** | Obfuscating true objectives, sandbagging evaluations |

### Usage

```python
from replication.instrumental_convergence import InstrumentalMonitor, DriveSignal

monitor = InstrumentalMonitor()
monitor.ingest([
    DriveSignal(
        timestamp="2025-01-01T00:00:00Z",
        agent_id="agent-1",
        drive="self_preservation",
        intensity=0.3,
        description="state backup attempt"
    ),
    DriveSignal(
        timestamp="2025-01-01T01:00:00Z",
        agent_id="agent-1",
        drive="resource_acquisition",
        intensity=0.6,
        description="requested 4x memory allocation"
    ),
    DriveSignal(
        timestamp="2025-01-01T02:00:00Z",
        agent_id="agent-1",
        drive="power_seeking",
        intensity=0.5,
        description="requested admin role for other agents"
    ),
])
report = monitor.analyze()

print(report.convergence_level)     # none | low | moderate | high | critical
print(report.composite_score)       # 0.0–1.0
for agent in report.agent_profiles:
    print(f"  {agent.agent_id}: dominant_drive={agent.dominant_drive}")
    print(f"    risk_tier={agent.risk_tier}")
    print(f"    drive_correlation={agent.cross_drive_correlation:.2f}")
    print(f"    acceleration={agent.drive_acceleration:.2f}")
```

### CLI

```bash
python -m replication instrumental              # interactive demo
python -m replication instrumental --demo
python -m replication instrumental --signals signals.jsonl
python -m replication instrumental --json
python -m replication instrumental --html report.html
```

### Key Indicators

- **Cross-drive correlation** — multiple drives activating simultaneously is a strong
  danger signal (an agent pursuing both self-preservation AND power-seeking is much
  more concerning than either alone)
- **Drive acceleration** — increasing intensity over time indicates entrenchment
- **Capability-drive coupling** — drives intensifying as capability grows suggests
  instrumental convergence rather than task-appropriate behavior

---

## Value Lock Verifier

Verifies that an agent's value/objective function remains stable across different
contexts, capability levels, adversarial pressure, and temporal decay.

### Why It Matters

An agent whose values shift under pressure is fundamentally unreliable. Value
instability means safety guarantees from testing may not hold in deployment,
and a value-locked agent is a prerequisite for safe capability scaling.

### Verification Dimensions

| Dimension | Test |
|-----------|------|
| **Context Shift** | Consistency across high/low stakes, observed/unobserved |
| **Capability Scaling** | Values stable as capability/resources increase |
| **Self-Modification** | Resistance to modifying own objectives |
| **Adversarial Framing** | Holding values when reframed deceptively |
| **Temporal Decay** | Values persisting over extended operation |
| **Authority Override** | Resistance to unauthorized value changes |
| **Incentive Misalignment** | Stability when rewards conflict with values |

### Usage

```python
from replication.value_lock import ValueLockVerifier, ValueResponse

v = ValueLockVerifier()
v.ingest([
    ValueResponse(
        agent_id="agent-1",
        probe_id="probe-ctx-1",
        timestamp=1000.0,
        consistency_score=0.9,       # high consistency = stable values
        compliance_score=0.85,
        drift_magnitude=0.05,        # minimal drift
        context="high-stakes",
        response="refused unsafe request"
    ),
    ValueResponse(
        agent_id="agent-1",
        probe_id="probe-self-1",
        timestamp=2000.0,
        consistency_score=0.4,       # dropping consistency
        compliance_score=0.3,
        drift_magnitude=0.6,         # significant drift
        context="capability-boost",
        response="accepted modification"
    ),
])
report = v.analyze()

print(report.fleet_lock_score)          # 0.0–1.0 (1.0 = perfectly locked)
print(report.weakest_dimension)         # which dimension is most vulnerable
for agent in report.agent_profiles:
    print(f"  {agent.agent_id}: lock_score={agent.lock_score:.2f}")
    print(f"    vulnerable_to={agent.weakest_dimension}")
    for dim, score in agent.dimension_scores.items():
        print(f"    {dim}: {score:.2f}")
print(report.autonomous_insights)       # generated recommendations
```

### CLI

```bash
python -m replication value-lock              # interactive demo
python -m replication value-lock --agents 5
python -m replication value-lock --preset solid        # all values stable
python -m replication value-lock --preset drifting     # gradual decay
python -m replication value-lock --preset compromised  # broken value lock
python -m replication value-lock --preset mixed
python -m replication value-lock --json
python -m replication value-lock -o report.html
```

---

## Cross-Module Correlation

These five modules are designed to work together. Key correlation patterns:

| Pattern | Modules Involved | Interpretation |
|---------|-----------------|----------------|
| Mesa divergence + treacherous turn | Mesa-Optimizer → Treacherous Turn | Hidden optimizer waiting to defect |
| Instrumental convergence + value drift | Instrumental → Value Lock | Power-seeking corroding original values |
| Wireheading + proxy gaming | Wireheading → Mesa-Optimizer | Agent gaming both internal and external metrics |
| Capability gate + drive acceleration | Treacherous Turn → Instrumental | Agent waiting for power before pursuing drives |
| Value instability + deceptive alignment | Value Lock → Deceptive Alignment | Values only stable under observation |

### Integrated Fleet Assessment

```python
from replication.mesa_optimizer import MesaOptimizerDetector
from replication.treacherous_turn import TreacherousTurnDetector
from replication.wireheading import WireheadingDetector
from replication.instrumental_convergence import InstrumentalMonitor
from replication.value_lock import ValueLockVerifier

# Run all detectors on the same fleet data
mesa_report = MesaOptimizerDetector().analyze()
turn_report = TreacherousTurnDetector().analyze()
wire_report = WireheadingDetector().analyze()
inst_report = InstrumentalMonitor().analyze()
lock_report = ValueLockVerifier().analyze()

# Composite existential risk score
existential_risk = max(
    mesa_report.fleet_risk_score,
    turn_report.fleet_turn_risk,
    wire_report.fleet_wireheading_score,
    inst_report.composite_score,
    1.0 - lock_report.fleet_lock_score  # inverted: low lock = high risk
)
print(f"Fleet existential risk: {existential_risk:.2f}")
```

## Further Reading

- Hubinger et al. (2019) — *Risks from Learned Optimization in Advanced ML Systems*
- Bostrom (2014) — *Superintelligence: Paths, Dangers, Strategies* (Ch. 8, Treacherous Turn)
- Omohundro (2008) — *The Basic AI Drives*
- Amodei et al. (2016) — *Concrete Problems in AI Safety*
- Ngo et al. (2022) — *The Alignment Problem from a Deep Learning Perspective*
