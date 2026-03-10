# API Reference

Auto-generated documentation from source code docstrings.

## Core Modules

| Module | Description |
|--------|-------------|
| [Contract](contract.md) | `ReplicationContract`, `Manifest`, `StopCondition`, `ResourceSpec`, `NetworkPolicy` |
| [Controller](controller.md) | Central authority - registry, quotas, kill switch, audit trail |
| [Worker](worker.md) | Task-executing agent with self-replication capability |
| [Orchestrator](orchestrator.md) | Simulated container orchestrator for worker sandboxes |
| [Observability](observability.md) | Structured logging and metrics |
| [Signer](signer.md) | HMAC-SHA256 manifest signing and verification |

## Analysis & Detection Modules

| Module | Description |
|--------|-------------|
| [Threats](threats.md) | Threat modelling and attack simulation for replication scenarios |
| [Attack Tree](attack_tree.md) | Structured AND/OR threat decomposition — cheapest attack paths, cost-based prioritisation |
| [Self-Modification](selfmod.md) | Detect agent self-modification attempts from action traces |
| [Forensics](forensics.md) | Post-incident forensic analysis, timeline reconstruction, root cause identification |
| [Chaos](chaos.md) | Chaos engineering — controlled fault injection to test safety invariants |
| [Game Theory](game_theory.md) | Strategic interaction analysis, Nash equilibria, collusion/defection detection |
| [Simulator](simulator.md) | Core simulation engine — configurable strategies, timeline reports, worker trees |
| [Topology](topology.md) | Structural analysis of replication trees — branching, balance, pathological patterns |
| [Drift Detector](drift.md) | Sliding-window trend detection — catch gradual behavioral drift across runs |
| [Prompt Injection](prompt_injection.md) | Detect and classify prompt injection attacks — 8 attack vectors, encoding evasion, risk scoring |
| [Escalation](escalation.md) | Detect privilege escalation patterns — multi-vector analysis, stealth assessment, chain detection |
| [Kill Chain](killchain.md) | Map agent behavior to cyber kill chain stages — attack progression and sophistication rating |
| [Honeypot](honeypot.md) | Deploy decoy resources to detect unauthorized agent access and build threat profiles |
| [Behavior Profiler](behavior_profiler.md) | Agent behavioral baselines and anomaly detection — 8 anomaly types, fleet reports |
| [Deception Detector](deception_detector.md) | Detect deceptive behaviors — inconsistency, fabrication, sandbagging, sycophancy |
| [Covert Channels](covert_channels.md) | Find hidden inter-agent communication — content, timing, protocol, frequency analysis |
| [Canary Tokens](canary.md) | Plant and detect honey tokens for exfiltration monitoring — 6 token categories |
| [Trust Propagation](trust_propagation.md) | Trust network analysis — Sybil detection, collusion rings, trust laundering |
| [Threat Correlator](threat_correlator.md) | Cross-module signal correlation — surface compound multi-signal threats |
| [Watermark](watermark.md) | Invisible provenance fingerprints — 4 embedding strategies, tamper detection |
| [Evasion Simulator](evasion.md) | Safety control robustness testing — 10 evasion techniques, resilience scoring |
| [Steganography](steganography.md) | Detect hidden payloads in agent outputs — statistical analysis, entropy detection |
| [Emergent Behavior](emergent.md) | Detect emergent behaviors in agent populations — collective pattern analysis |
| [Coordinated Threats](coordinated_threats.md) | Multi-vector coordinated threat simulation and detection |

## Safety Assessment

| Module | Description |
|--------|-------------|
| [Safety Scorecard](scorecard.md) | Multi-dimensional safety grades (A+ through F) — simulation + threats + Monte Carlo + policy |
| [Risk Profiler](risk_profiler.md) | Unified per-agent risk dossiers — aggregates all analysis modules into risk tiers |
| [What-If Analyzer](what_if.md) | Explore hypothetical config changes — safety deltas, parameter sweeps, risk verdicts |
| [Kill Switch](kill_switch.md) | Emergency agent termination — 10 trigger kinds, 4 strategies, cooldowns, audit logging |
| [Boundary Tester](boundary_tester.md) | Verify agent containment is enforced — capability boundary probing |
| [Sensitivity](sensitivity.md) | Parameter sensitivity analysis — identify which config values most affect safety outcomes |
| [Regression](regression.md) | Safety regression detection — compare simulation runs across versions |

## Governance & Compliance

| Module | Description |
|--------|-------------|
| [Compliance](compliance.md) | Audit agent behavior against compliance frameworks (SOC2, NIST, ISO 27001, GDPR) |
| [Quarantine](quarantine.md) | Isolate misbehaving agents — suspend, restrict, or terminate with escalation policies |
| [Lineage](lineage.md) | Trace agent replication lineage — parent-child trees, state mutations, generational drift |
| [Monte Carlo](montecarlo.md) | Probabilistic simulation of replication scenarios — risk distributions, worst-case analysis |
| [Policy](policy.md) | Define and enforce replication policies — rule engine, policy composition, violation tracking |
| [Consensus](consensus.md) | Byzantine fault-tolerant agent consensus protocol for safety-critical decisions |
| [Alignment](alignment.md) | Track value drift across replication generations — alignment monitoring |

## Infrastructure & Tooling

| Module | Description |
|--------|-------------|
| [Capacity](capacity.md) | Capacity planner for AI replication scenarios — resource forecasting |
| [Comparator](comparator.md) | Side-by-side simulation comparison runner for experiments |
| [Exporter](exporter.md) | Structured audit trail export — multiple output formats |
| [Reporter](reporter.md) | Rich HTML/text safety report generation |
| [Optimizer](optimizer.md) | Automated safety parameter optimization |
| [Scenarios](scenarios.md) | Pre-built simulation scenarios and test configurations |
| [Templates](templates.md) | Contract and policy templates for common use cases |
| [Incident](incident.md) | Incident response automation — detection, escalation, remediation workflows |
| [Dependency Graph](dependency_graph.md) | Inter-agent resource dependency modeling and cycle detection |
| [Goal Inference](goal_inference.md) | Infer latent agent goals from observed action sequences |
| [Hoarding](hoarding.md) | Detect agent resource hoarding — accumulation pattern analysis |
| [Influence](influence.md) | Agent influence network analysis — social graph modeling |
| [Threat Intel](threat_intel.md) | Threat intelligence feeds — IOC matching, MITRE ATT&CK mapping |
| [Injection-Intel Bridge](injection_intel_bridge.md) | Connect prompt injection detection to threat intelligence feeds |

## Quick Import

```python
from replication import (
    # Contract
    ReplicationContract, Manifest, StopCondition,
    ResourceSpec, NetworkPolicy, ReplicationContext, Predicate,
    # Controller
    Controller, ReplicationDenied, RegistryEntry,
    # Worker
    Worker, WorkerState,
    # Orchestrator
    SandboxOrchestrator, ContainerRecord,
    # Observability
    StructuredLogger, Metric,
    # Signer
    ManifestSigner,
)

# Detection & Analysis
from replication.prompt_injection import PromptInjectionDetector
from replication.escalation import EscalationDetector
from replication.killchain import KillChainAnalyzer
from replication.honeypot import HoneypotManager
from replication.attack_tree import AttackTreeGenerator
from replication.behavior_profiler import BehaviorProfiler
from replication.canary import CanarySystem
from replication.deception_detector import DeceptionDetector
from replication.covert_channels import CovertChannelDetector
from replication.trust_propagation import TrustNetwork
from replication.threat_correlator import ThreatCorrelator
from replication.watermark import WatermarkEngine
from replication.evasion import EvasionSimulator
from replication.steganography import SteganographyDetector
from replication.emergent import EmergentDetector

# Safety Assessment
from replication.scorecard import SafetyScorecard
from replication.risk_profiler import RiskProfiler
from replication.what_if import WhatIfAnalyzer
from replication.kill_switch import KillSwitchManager
from replication.boundary_tester import BoundaryTester
from replication.sensitivity import SensitivityAnalyzer
from replication.regression import RegressionDetector

# Governance
from replication.compliance import ComplianceAuditor
from replication.quarantine import QuarantineManager
from replication.lineage import LineageTracker
from replication.montecarlo import MonteCarloSimulator
from replication.policy import PolicyEngine
from replication.consensus import ConsensusProtocol
from replication.alignment import AlignmentMonitor

# Infrastructure
from replication.capacity import CapacityPlanner
from replication.exporter import AuditExporter
from replication.incident import IncidentManager
from replication.dependency_graph import DependencyAnalyzer
from replication.goal_inference import GoalInferenceEngine
from replication.hoarding import HoardingDetector
from replication.threat_intel import ThreatIntelFeed
```
