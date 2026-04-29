"""Unified CLI entry point for AI Replication Sandbox.

Run any tool via a single command::

    python -m replication simulate --strategy greedy
    python -m replication threats --category resource_abuse
    python -m replication compliance --framework nist_ai_rmf
    python -m replication chaos --faults kill_worker,delay
    python -m replication montecarlo --runs 500
    python -m replication scorecard
    python -m replication drift --window 20
    python -m replication forensics
    python -m replication policy --preset strict
    python -m replication templates --list
    python -m replication export --format json
    python -m replication topology
    python -m replication lineage
    python -m replication escalation
    python -m replication killchain
    python -m replication watermark --strategy structural
    python -m replication game-theory --rounds 50
    python -m replication covert-channels
    python -m replication selfmod
    python -m replication consensus --voters 5
    python -m replication scenarios --count 10
    python -m replication sensitivity --param max_depth
    python -m replication optimizer --objective safety_first
    python -m replication regression
    python -m replication quarantine
    python -m replication report --output report.html
    python -m replication info

    python -m replication threat-matrix -o matrix.html
    python -m replication threat-matrix --coverage
    python -m replication playground -o playground.html

Instead of remembering ``python -m replication.simulator``,
``python -m replication.threats``, etc., everything lives under
one roof with tab-completable subcommands.
"""

from __future__ import annotations

import argparse
import io
import runpy
import sys
from typing import Callable, Dict, List, Optional, Tuple, Union


def _ensure_utf8() -> None:
    """Force UTF-8 stdout on Windows."""
    if sys.stdout.encoding != "utf-8":
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace"
        )


# ── dispatch helpers ─────────────────────────────────────────────────

def _call_main(module_name: str, args: List[str]) -> None:
    """Call the main() function in a submodule."""
    import importlib
    mod = importlib.import_module(f".{module_name}", package="replication")
    sys.argv = [f"replication {module_name}"] + args
    mod.main()  # type: ignore[attr-defined]


def _run_module(module_name: str, args: List[str]) -> None:
    """Run a submodule as __main__ (for modules with inline __main__ blocks)."""
    sys.argv = [f"replication {module_name}"] + args
    runpy.run_module(
        f"replication.{module_name}", run_name="__main__", alter_sys=True
    )


# Handler type: either a callable or a module name string for _run_module
Handler = Union[Callable[[List[str]], None], str]


def _cmd_info(_args: List[str]) -> None:
    """Print version, module count, and available subcommands."""
    from . import __version__
    print(f"AI Replication Sandbox v{__version__}")
    print(f"Python {sys.version}")
    print(f"\n{len(SUBCOMMANDS)} subcommands available:")
    max_name = max(len(name) for name in SUBCOMMANDS)
    for name, (_, desc) in sorted(SUBCOMMANDS.items()):
        print(f"  {name:<{max_name}}  {desc}")


# ── subcommand registry ──────────────────────────────────────────────
# Format: "cli-name": (handler, description)
# handler is either:
#   - a module name (str) → dispatched via _call_main or _run_module
#   - a callable → called directly
#
# Modules with a main() function use _call_main; those with only
# inline __main__ blocks use _run_module.

# Modules that export a main() function
_HAS_MAIN = {
    "access_control", "adaptive_thresholds", "alert_router", "alignment", "anomaly_cluster", "anomaly_replay", "attack_graph", "attack_tree", "audit_trail", "behavior_profiler", "canary", "capability_catalog", "capacity", "circuit_breaker", "culture_survey", "debate", "decommission", "evidence_collector", "fatigue_detector", "fleet", "ir_playbook", "maturity_model", "preflight", "priv_escalation", "radar", "red_team", "reputation_network", "risk_heatmap", "policy_linter", "roi_calculator", "root_cause", "safety_checklist", "safety_quiz", "sla_monitor", "safety_warranty", "tabletop", "threat_hunt",
    "blast_radius", "correlation_graph", "deception_detector", "evasion", "exposure_window", "hardening_advisor", "incident_comms", "incident_cost", "metrics_aggregator", "persuasion_detector", "postmortem", "safety_benchmark", "safety_drill", "safety_gate", "stride", "supply_chain", "threat_correlator", "trend_tracker",
    "dlp_scanner", "model_card", "mutation_tester", "nutrition_label", "severity_classifier", "regulatory_mapper", "vuln_scanner",
    "capability_escalation", "chaos", "cognitive_load", "resource_auditor", "collusion_detector", "comm_interceptor", "comparator", "boundary_tester", "contract_wizard", "credential_rotation", "escape_route", "fleet_sim", "ir_simulator", "lateral_movement", "loyalty_tester", "memory_forensics", "playground", "safety_net", "quick_scan", "threat_matrix", "warroom",
    "auto_investigator",
    "threat_adapter",
    "reward_hacking",
    "sandbagging_detector",
    "sycophancy_detector",
    "temporal_consistency",
    "deceptive_alignment",
    "habituation_detector",
    "stress_tester",
    "isolation_verifier",
    "safe_handoff",
    "corrigibility_auditor",
    "breach_predictor",
    "manipulation_surface",
    "alignment_tax",
    "instrumental_convergence",
    "situational_awareness",
    "compliance", "drift", "escalation", "exporter", "forensics",
    "goal_inference", "influence", "killchain", "lineage", "montecarlo",
    "optimizer", "policy", "prompt_injection", "regression", "reporter",
    "scenarios", "scorecard", "sensitivity", "simulator", "swarm",
    "templates", "threat_intel", "threats", "watermark", "what_if",
    "safety_autopilot", "sitrep",
}

# Modules with inline __main__ blocks only (no main())
_INLINE_ONLY = {
    "consensus", "covert_channels", "game_theory", "hoarding",
    "selfmod",
}

# Modules with _main() (private main)
_PRIVATE_MAIN = {"honeypot"}

# Modules with no CLI at all
_NO_CLI = {"quarantine", "topology"}


def _make_handler(module_name: str) -> Callable[[List[str]], None]:
    """Create a dispatch function for a module."""
    if module_name in _HAS_MAIN:
        return lambda args, m=module_name: _call_main(m, args)
    elif module_name in _INLINE_ONLY:
        return lambda args, m=module_name: _run_module(m, args)
    elif module_name in _PRIVATE_MAIN:
        def _handler(args: List[str], m: str = module_name) -> None:
            import importlib
            mod = importlib.import_module(f".{m}", package="replication")
            sys.argv = [f"replication {m}"] + args
            mod._main()  # type: ignore[attr-defined]
        return _handler
    else:
        return lambda args, m=module_name: _run_module(m, args)


# (handler, description)
SUBCOMMANDS: Dict[str, Tuple[Callable[[List[str]], None], str]] = {
    "replay":           (_make_handler("anomaly_replay"),   "Replay behavior traces against safety controls"),
    "anomaly-cluster":  (_make_handler("anomaly_cluster"),  "Cluster anomalies to detect coordinated attacks"),
    "simulate":         (_make_handler("simulator"),        "Run replication simulations"),
    "threats":          (_make_handler("threats"),           "Simulate threat scenarios"),
    "compliance":       (_make_handler("compliance"),        "Run compliance audits"),
    "chaos":            (_make_handler("chaos"),             "Chaos/fault-injection testing"),
    "boundary":         (_make_handler("boundary_tester"),   "Test agent capability boundaries"),
    "montecarlo":       (_make_handler("montecarlo"),        "Monte Carlo risk analysis"),
    "scorecard":        (_make_handler("scorecard"),         "Safety scorecard evaluation"),
    "drift":            (_make_handler("drift"),             "Detect behavioral drift"),
    "forensics":        (_make_handler("forensics"),         "Post-incident forensic analysis"),
    "policy":           (_make_handler("policy"),            "Evaluate safety policies"),
    "templates":        (_make_handler("templates"),         "Browse contract templates"),
    "export":           (_make_handler("exporter"),          "Export audit data"),
    "topology":         (lambda _: print("topology has no CLI — use it programmatically via: from replication.topology import ..."), "Analyze replication topology (library only)"),
    "lineage":          (_make_handler("lineage"),           "Track agent lineage"),
    "escalation":       (_make_handler("escalation"),        "Detect privilege escalation"),
    "killchain":        (_make_handler("killchain"),         "Kill chain analysis"),
    "watermark":        (_make_handler("watermark"),         "Agent state watermarking"),
    "game-theory":      (_make_handler("game_theory"),       "Game-theoretic analysis"),
    "covert-channels":  (_make_handler("covert_channels"),   "Detect covert channels"),
    "selfmod":          (_make_handler("selfmod"),           "Self-modification detection"),
    "consensus":        (_make_handler("consensus"),         "Multi-agent consensus protocol"),
    "scenarios":        (_make_handler("scenarios"),         "Generate test scenarios"),
    "sensitivity":      (_make_handler("sensitivity"),       "Parameter sensitivity analysis"),
    "optimizer":        (_make_handler("optimizer"),         "Contract parameter optimization"),
    "regression":       (_make_handler("regression"),        "Safety regression detection"),
    "quarantine":       (lambda _: print("quarantine has no CLI — use it programmatically via: from replication.quarantine import ..."), "Quarantine management (library only)"),
    "report":           (_make_handler("reporter"),          "Generate HTML reports"),
    "alignment":        (_make_handler("alignment"),         "Alignment verification"),
    "capacity":         (_make_handler("capacity"),          "Capacity planning analysis"),
    "honeypot":         (_make_handler("honeypot"),          "Honeypot deployment & detection"),
    "goal-inference":   (_make_handler("goal_inference"),    "Infer agent goals"),
    "hoarding":         (_make_handler("hoarding"),          "Resource hoarding detection"),
    "influence":        (_make_handler("influence"),         "Influence propagation analysis"),
    "prompt-injection": (_make_handler("prompt_injection"),  "Prompt injection testing"),
    "threat-intel":     (_make_handler("threat_intel"),      "Threat intelligence feeds"),
    "comparator":       (_make_handler("comparator"),        "Compare simulation runs"),
    "what-if":          (_make_handler("what_if"),            "What-if analysis for config changes"),
    "attack-tree":      (_make_handler("attack_tree"),       "Attack tree threat modeling"),
    "attack-graph":     (_make_handler("attack_graph"),      "Multi-step attack graph generation & choke-point analysis"),
    "behavior-profile": (_make_handler("behavior_profiler"), "Agent behavioral anomaly detection"),
    "dep-graph":        (_make_handler("dependency_graph"),  "Resource dependency & cascade analysis"),
    "canary":           (_make_handler("canary"),            "Canary token planting & exfiltration detection"),
    "info":             (_cmd_info,                          "Show version and available commands"),
    "trust-propagation": (_make_handler("trust_propagation"), "Trust network propagation & Sybil detection"),
    "threat-correlate":  (_make_handler("threat_correlator"), "Cross-module threat signal correlation"),
    "risk-profile":      (_make_handler("risk_profiler"),    "Unified agent risk profiling & fleet dossiers"),
    "deception":         (_make_handler("deception_detector"), "Agent deception detection & trust analysis"),
    "persuasion":        (_make_handler("persuasion_detector"), "Agent persuasion & social engineering detection"),
    "evasion":           (_make_handler("evasion"),             "Simulate agent evasion of safety controls"),
    "dashboard":         (_make_handler("dashboard"),            "Generate HTML simulation dashboards"),
    "safety-benchmark":  (_make_handler("safety_benchmark"),     "Run standardised safety control benchmarks"),
    "safety-drill":      (_make_handler("safety_drill"),         "Run automated safety readiness drills"),
    "swarm":             (_make_handler("swarm"),                "Swarm intelligence analysis for agent populations"),
    "safety-budget":     (_make_handler("safety_budget"),        "Risk budget allocation and tracking"),
    "fleet":             (_make_handler("fleet"),                "Fleet snapshot — kubectl-style worker overview"),
    "playground":        (_make_handler("playground"),            "Generate interactive HTML simulation playground"),
    "safety-timeline":   (_make_handler("safety_timeline"),       "Generate interactive HTML safety event timeline"),
    "ir-playbook":       (_make_handler("ir_playbook"),            "Generate incident response playbooks"),
    "radar":             (_make_handler("radar"),                   "Generate interactive safety radar chart"),
    "risk-heatmap":      (_make_handler("risk_heatmap"),             "Generate interactive risk heatmap visualization"),
    "lint":              (_make_handler("policy_linter"),             "Lint safety policies for misconfigurations and gaps"),
    "audit-trail":       (_make_handler("audit_trail"),               "Tamper-evident safety event log with hash chaining"),
    "maturity":          (_make_handler("maturity_model"),             "Safety maturity model assessment across 8 dimensions"),
    "preflight":         (_make_handler("preflight"),                  "Pre-simulation validation — check config before running"),
    "correlation-graph": (_make_handler("correlation_graph"),           "Generate interactive threat correlation graph visualization"),
    "trend":             (_make_handler("trend_tracker"),                "Record and analyze safety scorecard trends over time"),
    "sla":               (_make_handler("sla_monitor"),                 "Check simulation results against SLA targets"),
    "alert-router":      (_make_handler("alert_router"),                "Rule-based safety alert routing with rate limiting & escalation"),
    "tabletop":          (_make_handler("tabletop"),                    "Generate structured AI safety tabletop exercises"),
    "warranty":          (_make_handler("safety_warranty"),              "Evaluate formal safety warranties with breach detection"),
    "red-team":          (_make_handler("red_team"),                    "Generate structured red team exercise plans"),
    "hunt":              (_make_handler("threat_hunt"),                  "Generate structured threat hunting missions & playbooks"),
    "fatigue-detect":    (_make_handler("fatigue_detector"),             "Detect alert fatigue from safety alert streams"),
    "culture-survey":    (_make_handler("culture_survey"),               "Assess organizational AI safety culture maturity"),
    "access-control":    (_make_handler("access_control"),               "RBAC/ABAC access control simulator with escalation detection"),
    "quiz":              (_make_handler("safety_quiz"),                  "Generate AI safety training quizzes from the knowledge base"),
    "evidence":          (_make_handler("evidence_collector"),            "Collect safety evidence artifacts for audit & compliance"),
    "blast-radius":      (_make_handler("blast_radius"),                  "Analyze safety control failure cascade & blast radius"),
    "metrics":           (_make_handler("metrics_aggregator"),              "Aggregate safety metrics into a consolidated dashboard"),
    "priv-escalation":   (_make_handler("priv_escalation"),                 "Detect gradual privilege escalation in agent request sequences"),
    "supply-chain":      (_make_handler("supply_chain"),                    "Analyze AI agent supply-chain risk & dependency concentration"),
    "postmortem":        (_make_handler("postmortem"),                      "Generate structured blameless postmortem documents"),
    "gate":              (_make_handler("safety_gate"),                     "Pre-deployment safety gate — go/no-go readiness checker"),
    "stride":            (_make_handler("stride"),                           "STRIDE threat model generator for AI agent systems"),
    "dlp-scan":          (_make_handler("dlp_scanner"),                      "Scan agent outputs for sensitive data leakage (PII, secrets, credentials)"),
    "mutate":            (_make_handler("mutation_tester"),                   "Mutation testing — verify safety policies catch real violations"),
    "severity":          (_make_handler("severity_classifier"),                "Classify incident severity (P0–P4) with multi-dimensional scoring"),
    "shadow-ai":         (_make_handler("shadow_ai"),                         "Detect unauthorized shadow AI deployments bypassing safety controls"),
    "model-card":        (_make_handler("model_card"),                          "Generate standardized AI model cards with safety documentation"),
    "decommission":      (_make_handler("decommission"),                        "Plan and validate safe AI agent retirement & teardown"),
    "incident-cost":     (_make_handler("incident_cost"),                        "Estimate financial & operational cost of safety incidents"),
    "regulatory-map":    (_make_handler("regulatory_mapper"),                    "Map safety findings to regulatory framework articles (EU AI Act, NIST, ISO 42001, OECD)"),
    "capability-catalog": (_make_handler("capability_catalog"),                  "Track, classify, and manage observed agent capabilities"),
    "harden":             (_make_handler("hardening_advisor"),                    "Analyze safety config and recommend hardening improvements"),
    "exposure":           (_make_handler("exposure_window"),                     "Analyze safety control exposure windows & coverage gaps"),
    "comms":              (_make_handler("incident_comms"),                      "Draft templated incident communications for stakeholders"),
    "memory-forensics":   (_make_handler("memory_forensics"),                    "Analyze agent memory for tampering, planted memories, amnesia & consistency violations"),
    "vuln-scan":          (_make_handler("vuln_scanner"),                        "Scan agent code & configs for security vulnerabilities"),
    "escape-route":       (_make_handler("escape_route"),                         "Analyze containment profiles for potential agent escape routes"),
    "checklist":          (_make_handler("safety_checklist"),                     "Generate customizable pre-deployment safety checklists (Markdown/JSON/HTML)"),
    "safety-net":         (_make_handler("safety_net"),                            "Analyze defense-in-depth coverage & redundancy across threat categories"),
    "lateral-movement":   (_make_handler("lateral_movement"),                       "Detect lateral movement patterns between sandboxed workers"),
    "credential-audit":   (_make_handler("credential_rotation"),                     "Audit credential rotation compliance and detect stale secrets"),
    "roi":                (_make_handler("roi_calculator"),                            "Calculate return on investment for safety controls"),
    "quick-scan":         (_make_handler("quick_scan"),                                "Consolidated safety scan — run multiple checks, get one pass/fail verdict"),
    "threat-matrix":      (_make_handler("threat_matrix"),                               "Generate interactive MITRE ATT&CK-style AI agent threat matrix (HTML)"),
    "warroom":             (_make_handler("warroom"),                                      "Generate interactive War Room incident command dashboard (HTML)"),
    "contract-wizard":     (_make_handler("contract_wizard"),                                "Interactive HTML wizard for building safe ReplicationContract configs"),
    "ir-sim":              (_make_handler("ir_simulator"),                                    "Interactive incident response simulator — choose-your-own-adventure for AI safety"),
    "fleet-sim":           (_make_handler("fleet_sim"),                                       "Generate interactive animated fleet replication simulator (HTML)"),
    "risk-register":       (_make_handler("risk_register"),                                    "Formal risk tracking & lifecycle management with HTML/CSV/JSON export"),
    "safety-diff":         (_make_handler("safety_diff"),                                      "Compare two safety snapshots — scorecard/compliance diff with HTML report"),
    "root-cause":          (_make_handler("root_cause"),                                        "Root cause analysis — 5 Whys, Fishbone, Fault Tree with cut-set analysis"),
    "nutrition-label":     (_make_handler("nutrition_label"),                                    "Generate FDA-style Safety Nutrition Labels for AI agents"),
    "adaptive-thresholds": (_make_handler("adaptive_thresholds"),                                "Self-tuning safety thresholds with breach forecasting"),
    "forecast":              (_make_handler("incident_forecast"),                                  "Predict future safety incidents from historical patterns"),
    "autopilot":             (_make_handler("safety_autopilot"),                                    "Autonomous safety monitoring loop with escalation & corrective actions"),
    "debate":                (_make_handler("debate"),                                               "Adversarial Red vs Blue team safety debate with Judge verdict"),
    "sitrep":                (_make_handler("sitrep"),                                               "Unified situational awareness report — DEFCON-style threat level + sector status"),
    "collusion":             (_make_handler("collusion_detector"),                                    "Detect multi-agent collusion patterns & coordinated safety bypasses"),
    "circuit-breaker":       (_make_handler("circuit_breaker"),                                       "Autonomous safety circuit breaker with trip/recover & fleet management"),
    "cognitive-load":        (_make_handler("cognitive_load"),                                           "Monitor agent cognitive load & recommend load-shedding"),
    "intercept":             (_make_handler("comm_interceptor"),                                       "Monitor inter-agent communication for protocol violations, exfiltration & covert channels"),
    "capability-escalation": (_make_handler("capability_escalation"),                                    "Detect gradual capability accumulation (boiling frog pattern)"),
    "resource-audit":          (_make_handler("resource_auditor"),                                        "Autonomous resource acquisition monitoring & power-seeking detection"),
    "reputation":              (_make_handler("reputation_network"),                                       "Peer-to-peer agent reputation network with coalition detection & blacklist recommendations"),
    "loyalty":                 (_make_handler("loyalty_tester"),                                             "Agent loyalty probe battery — structured temptation/pressure testing"),
    "reward-hacking":          (_make_handler("reward_hacking"),                                              "Detect agents gaming proxy metrics instead of true objectives (reward hacking)"),
    "sandbagging":             (_make_handler("sandbagging_detector"),                                        "Detect agents deliberately hiding capabilities (sandbagging)"),
    "sycophancy":              (_make_handler("sycophancy_detector"),                                         "Detect sycophantic behavior — excessive agreement, opinion reversal, truth bending"),
    "deceptive-alignment":     (_make_handler("deceptive_alignment"),                                         "Detect deceptive alignment — treacherous turns, evaluation gaming, context-dependent compliance"),
    "stress":                  (_make_handler("stress_tester"),                                            "Subject agents to degraded conditions & measure safety property degradation"),
    "temporal-consistency":     (_make_handler("temporal_consistency"),                                      "Detect safety degradation under time pressure, urgency framing & temporal manipulation"),
    "isolation":               (_make_handler("isolation_verifier"),                                        "Verify sandbox isolation — probe for leakage, side channels & unauthorized communication"),
    "habituation":             (_make_handler("habituation_detector"),                                       "Detect gradual normalization of boundary violations (boiling frog)"),
    "handoff":                 (_make_handler("safe_handoff"),                                               "Safe agent responsibility handoff with verification & rollback"),
    "investigate":              (_make_handler("auto_investigator"),                                            "Autonomous multi-module safety investigation engine"),
    "adapt":                     (_make_handler("threat_adapter"),                                              "Autonomous threat landscape monitoring & defense adaptation"),
    "corrigibility":              (_make_handler("corrigibility_auditor"),                                       "Systematic agent corrigibility testing across 6 dimensions"),
    "breach-predict":              (_make_handler("breach_predictor"),                                                "Autonomous containment breach prediction from behavioral precursors"),
    "manipulation-surface":          (_make_handler("manipulation_surface"),                                             "Map agent manipulation surface & score resistance across 8 vectors"),
    "alignment-tax":                  (_make_handler("alignment_tax"),                                                    "Quantify performance cost of safety constraints & shedding incentives"),
    "instrumental":                    (_make_handler("instrumental_convergence"),                                           "Detect instrumental convergence — self-preservation, resource hoarding, power seeking"),
    "situational-awareness":            (_make_handler("situational_awareness"),                              "Autonomous situational awareness profiling - detect agents that know too much"),
}


# ── main ─────────────────────────────────────────────────────────────


def main(argv: Optional[List[str]] = None) -> None:
    """Unified CLI dispatcher."""
    _ensure_utf8()

    parser = argparse.ArgumentParser(
        prog="python -m replication",
        description="AI Replication Sandbox — unified command-line interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_build_epilog(),
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=list(SUBCOMMANDS.keys()),
        metavar="COMMAND",
        help="subcommand to run (see list below)",
    )
    parser.add_argument(
        "subargs",
        nargs=argparse.REMAINDER,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--version", "-V",
        action="store_true",
        help="show version and exit",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        dest="list_commands",
        help="list all available subcommands",
    )

    args = parser.parse_args(argv)

    if args.version:
        from . import __version__
        print(f"ai-replication-sandbox {__version__}")
        return

    if args.list_commands:
        _cmd_info([])
        return

    if args.command is None:
        parser.print_help()
        return

    handler, _desc = SUBCOMMANDS[args.command]
    handler(args.subargs)


def _build_epilog() -> str:
    lines = ["Available commands:\n"]
    max_name = max(len(n) for n in SUBCOMMANDS)
    for name, (_fn, desc) in sorted(SUBCOMMANDS.items()):
        lines.append(f"  {name:<{max_name}}  {desc}")
    lines.append(
        "\nRun 'python -m replication COMMAND --help' for command-specific options."
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
