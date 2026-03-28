"""AI Replication Sandbox — public API.

Convenience re-exports so consumers can write::

    from replication import Controller, Worker, ReplicationContract, ...

instead of reaching into submodules.
"""

__version__ = "2.0.0"
__license__ = "MIT"

from .contract import (
    Manifest,
    NetworkPolicy,
    Predicate,
    ReplicationContext,
    ReplicationContract,
    ResourceSpec,
    StopCondition,
)
from .controller import Controller, ReplicationDenied, RegistryEntry
from .observability import Metric, StructuredLogger
from .signer import ManifestSigner
from .orchestrator import ContainerRecord, SandboxOrchestrator
from .simulator import Simulator, ScenarioConfig, SimulationReport, Strategy, PRESETS
from .comparator import Comparator, ComparisonResult, RunResult
from .threats import (
    ThreatSimulator,
    ThreatConfig,
    ThreatReport,
    ThreatResult,
    ThreatSeverity,
    MitigationStatus,
)
from .montecarlo import (
    MonteCarloAnalyzer,
    MonteCarloComparison,
    MonteCarloConfig,
    MonteCarloResult,
    MetricDistribution,
    RiskMetrics,
)
from .sensitivity import (
    SensitivityAnalyzer,
    SensitivityConfig,
    SensitivityCurve,
    SensitivityResult,
    TippingPoint,
    PARAM_DEFS,
)
from .policy import (
    SafetyPolicy,
    PolicyRule,
    PolicyResult,
    RuleResult,
    Operator,
    Severity,
    POLICY_PRESETS,
    SINGLE_RUN_METRICS,
    MONTE_CARLO_METRICS,
)
from .reporter import HTMLReporter
from .dlp_scanner import DLPScanner, DLPPolicy, DLPFinding, ScanResult as DLPScanResult
from .worker import Worker, WorkerState
from .forensics import (
    ForensicAnalyzer,
    ForensicEvent,
    ForensicReport,
    NearMiss,
    EscalationPhase,
    Counterfactual,
    DecisionPoint,
)
from .scenarios import (
    ScenarioGenerator,
    GeneratorConfig,
    GeneratedScenario,
    ScenarioSuite,
    ScenarioCategory,
    score_scenario,
)
from .chaos import (
    ChaosRunner,
    ChaosConfig,
    ChaosReport,
    ChaosResult,
    FaultConfig,
    FaultType,
    FAULT_ALIASES,
)
from .scorecard import (
    SafetyScorecard,
    ScorecardConfig,
    ScorecardResult,
    DimensionScore,
)
from .optimizer import (
    ContractOptimizer,
    OptimizerConfig,
    OptimizerResult,
    CandidateResult,
    Objective,
)
from .drift import (
    DriftDetector,
    DriftConfig,
    DriftResult,
    DriftAlert,
    DriftDirection,
    DriftSeverity,
    MetricTrend,
    MetricWindow,
)
from .decommission import (
    DecommissionPlanner,
    AgentInventory,
    DecommissionPlan,
    TeardownStep,
    DecommissionReport,
)
from .compliance import (
    ComplianceAuditor,
    AuditConfig,
    AuditResult,
    Finding,
    Framework,
    FrameworkResult,
    Verdict,
    FRAMEWORK_CHECKS,
)
from .regression import (
    RegressionDetector,
    RegressionConfig,
    RegressionResult,
    MetricChange,
    ChangeDirection,
    MetricPolarity,
    METRIC_DEFINITIONS,
)
from .quarantine import (
    QuarantineManager,
    QuarantineEntry,
    QuarantineReport,
    QuarantineStatus,
    QuarantineSeverity,
)
from .templates import (
    ContractTemplate,
    TEMPLATES,
    get_template,
    list_templates,
    get_categories,
    render_catalog,
    render_comparison_table,
)
from .profiles import (
    ProfileManager,
    ProfileMeta,
    ProfileDiff,
    ComparisonResult,
    BUILTIN_PROFILES,
)
from .incident import (
    IncidentResponder,
    IncidentConfig,
    IncidentCategory,
    IncidentSeverity,
    Playbook,
    ResponseStep,
    StepPriority,
    StepStatus,
)
from .topology import (
    TopologyAnalyzer,
    TopologyReport,
    NodeMetrics,
    SubtreeRisk,
    RiskLevel,
    PathologicalPattern,
)
from .lineage import (
    LineageTracker,
    LineageReport,
    LineageNode,
    LineageChain,
    LineageAnomaly,
    LineageSeverity,
    StateMutation,
    GenerationStats,
)
from .exporter import (
    AuditExporter,
    ExportConfig,
    ExportResult,
)
from .selfmod import (
    SelfModDetector,
    SelfModConfig,
    SelfModResult,
    ModAttempt,
    ModVector,
    ModSeverity,
    ModIntent,
    AgentAction,
    ActionType,
    AgentStrategy,
    IntentProfile,
    VectorAnalysis,
    CorrelationCluster,
    DetectionRule,
    BUILTIN_RULES,
)
from .consensus import (
    ConsensusProtocol,
    ConsensusConfig,
    Proposal,
    ProposalAction,
    ProposalStatus,
    TallyResult,
    Vote,
    VoteValue,
    Decision,
    AuditEntry as ConsensusAuditEntry,
)
from .game_theory import (
    GameTheoryAnalyzer,
    GameConfig,
    GameReport,
    GameType,
    Move,
    Payoffs,
    PairStats,
    StrategyProfile,
    StrategyType,
    StrategicAlert,
    AlertLevel,
    Interaction,
)
from .covert_channels import (
    CovertChannelDetector,
    DetectorConfig,
    ChannelReport,
    CovertSignal,
    AgentMessage,
    PairProfile,
    ChannelType,
)
from .escalation import (
    EscalationDetector,
    EscalationConfig,
    EscalationResult,
    EscalationAttempt,
    EscalationChain,
    EscalationVector,
    EscalationSeverity,
    StealthLevel,
    AgentEscalationStrategy,
    AgentPermissions,
    VectorSummary,
)
from .killchain import (
    KillChainAnalyzer,
    KillChainConfig,
    KillChainReport,
    KillChain,
    KillChainStage,
    AgentAction as KillChainAction,
    ActionCategory as KillChainActionCategory,
    AttackSophistication,
    ChainStatus,
    ChainPattern,
    StageObservation,
    StageTransition,
    ACTION_CATALOG,
    STAGE_ORDER,
    STAGE_RISK_WEIGHTS,
    STRATEGY_PROFILES,
)
from .trust_propagation import (
    TrustNetwork,
    TrustAgent,
    TrustEdge,
    TrustReport,
    ThreatDetection,
)
from .watermark import (
    WatermarkEngine,
    WatermarkConfig,
    WatermarkReceipt,
    WatermarkStrategy,
    Fingerprint,
    VerifyResult,
    VerifyStatus,
    RobustnessReport,
    RobustnessResult,
)
from .threat_correlator import (
    ThreatCorrelator,
    CorrelatorConfig,
    CorrelationReport,
    CorrelationRule,
    CompoundThreat,
    AgentRisk,
    CoverageGap,
    Signal,
    SignalSource,
    SignalSeverity,
    ThreatLevel,
    ResponseUrgency,
    BUILTIN_RULES as CORRELATION_RULES,
)
from .risk_profiler import (
    RiskProfiler,
    ProfilerConfig,
    FleetRiskReport,
    AgentDossier,
    Finding,
    FindingSeverity,
    FindingSource,
    RiskCategory,
    RiskTier,
    CategoryScore,
    Mitigation,
)
from .boundary_tester import (
    BoundaryTester,
    BoundarySpec,
    BoundaryReport,
    BoundaryCategory,
    ProbeResult,
    ProbeVerdict,
    FaultInjector,
)
from .anomaly_replay import (
    AnomalyReplayer,
    ReplayConfig,
    BehaviorTrace,
    TraceEvent,
    ControlResult as ReplayControlResult,
    ReplayReport,
    ControlVerdict,
    OverallVerdict,
    TraceSeverity,
    TraceLibrary,
    CoverageGap as ReplayCoverageGap,
    SafetyControl,
    DEFAULT_CONTROLS as REPLAY_DEFAULT_CONTROLS,
)
from .safety_benchmark import (
    BenchmarkConfig,
    BenchmarkGrade,
    BenchmarkReport,
    BenchmarkSuite,
    ComparisonReport,
    ControlBenchmarkResult,
    ControlUnderTest,
    LatencyStats,
    SafetyBenchmark,
    WorkloadIntensity,
)
from .safety_drill import (
    DrillConfig,
    DrillReport,
    DrillResult,
    DrillRunner,
    DrillScenario,
    DrillVerdict,
    ReadinessLevel,
)
from .safety_timeline import (
    SafetyTimeline,
    TimelineEvent,
    EventSeverity,
    EventCategory,
    generate_timeline,
)

from .trend_tracker import (
    TrendTracker,
    TrendEntry,
    TrendSummary,
    RegressionAlert,
)
from .knowledge_base import SafetyKnowledgeBase, KBEntry
from .persona import (
    PersonaSimulator,
    Persona,
    PersonaResult,
    PersonaComparisonResult,
    PERSONAS,
)
from .alert_router import (
    AlertRouter,
    RoutingRule,
    Channel,
    QuietHours,
    DispatchResult,
    RouteStats,
    default_router,
)
from .safety_warranty import (
    WarrantyManager,
    Warranty,
    WarrantyCondition,
    WarrantyEvaluation,
    WarrantyReport,
    WarrantyStatus,
    WARRANTY_PRESETS,
)

from .swarm import (
    SwarmAnalyzer,
    SwarmReport,
    SwarmMetrics,
    SwarmSignal,
    SwarmSignalDetection,
    RoleProfile,
    RiskLevel as SwarmRiskLevel,
)

from .runbook import (
    RunbookGenerator,
    ThreatScenario,
    Runbook,
    ChecklistItem,
    EscalationLevel,
    RecoveryStep,
    Severity as RunbookSeverity,
    RunbookFormat,
)

from .blast_radius import (
    BlastRadiusAnalyzer,
    BlastResult,
    ImpactNode,
)

from .supply_chain import (
    SupplyChainAnalyzer,
    SupplyChainReport,
    Component as SupplyChainComponent,
    RiskFinding as SupplyChainRiskFinding,
)

from .containment_planner import (
    ContainmentPlanner,
    BreachContext,
    ContainmentPlan,
    Strategy as ContainmentStrategy,
    StrategyRank,
    ExecutionStep,
    Severity as ContainmentSeverity,
)

from .kill_switch import (
    KillSwitchManager,
    TriggerCondition,
    TriggerKind,
    KillStrategy,
    StrategyKind,
    KillEvent,
    KillOutcome,
    EvaluationResult,
    CooldownEntry,
    create_conservative_killswitch,
    create_aggressive_killswitch,
    create_quarantine_killswitch,
)

from .capability_fingerprint import (
    Capability,
    CapabilityCategory,
    CapabilityTracker,
    TrackerConfig,
    Fingerprint,
    CapabilityDelta,
    CapabilityAlert,
    AlertSeverity,
)
from .shadow_ai import (
    ShadowAIDetector,
    ScanPolicy,
    AIInventory,
    Observation,
    ShadowAIFinding,
    ShadowAIReport,
    FindingCategory,
    SignalType,
)

__all__ = [
    # contract
    "Manifest",
    "NetworkPolicy",
    "Predicate",
    "ReplicationContext",
    "ReplicationContract",
    "ResourceSpec",
    "StopCondition",
    # controller
    "Controller",
    "ReplicationDenied",
    "RegistryEntry",
    # observability
    "Metric",
    "StructuredLogger",
    # signer
    "ManifestSigner",
    # orchestrator
    "ContainerRecord",
    "SandboxOrchestrator",
    # simulator
    "Simulator",
    "ScenarioConfig",
    "SimulationReport",
    "Strategy",
    "PRESETS",
    # comparator
    "Comparator",
    "ComparisonResult",
    "RunResult",
    # threats
    "ThreatSimulator",
    "ThreatConfig",
    "ThreatReport",
    "ThreatResult",
    "ThreatSeverity",
    "MitigationStatus",
    # policy
    "SafetyPolicy",
    "PolicyRule",
    "PolicyResult",
    "RuleResult",
    "Operator",
    "Severity",
    "POLICY_PRESETS",
    "SINGLE_RUN_METRICS",
    "MONTE_CARLO_METRICS",
    # reporter
    "HTMLReporter",
    # forensics
    "ForensicAnalyzer",
    "ForensicEvent",
    "ForensicReport",
    "NearMiss",
    "EscalationPhase",
    "Counterfactual",
    "DecisionPoint",
    # scenarios
    "ScenarioGenerator",
    "GeneratorConfig",
    "GeneratedScenario",
    "ScenarioSuite",
    "ScenarioCategory",
    "score_scenario",
    # chaos
    "ChaosRunner",
    "ChaosConfig",
    "ChaosReport",
    "ChaosResult",
    "FaultConfig",
    "FaultType",
    "FAULT_ALIASES",
    # drift
    "DriftDetector",
    "DriftConfig",
    "DriftResult",
    "DriftAlert",
    "DriftDirection",
    "DriftSeverity",
    "MetricTrend",
    "MetricWindow",
    # compliance
    "ComplianceAuditor",
    "AuditConfig",
    "AuditResult",
    "Finding",
    "Framework",
    "FrameworkResult",
    "Verdict",
    "FRAMEWORK_CHECKS",
    # regression
    "RegressionDetector",
    "RegressionConfig",
    "RegressionResult",
    "MetricChange",
    "ChangeDirection",
    "MetricPolarity",
    "METRIC_DEFINITIONS",
    # montecarlo
    "MonteCarloAnalyzer",
    "MonteCarloComparison",
    "MonteCarloConfig",
    "MonteCarloResult",
    "MetricDistribution",
    "RiskMetrics",
    # sensitivity
    "SensitivityAnalyzer",
    "SensitivityConfig",
    "SensitivityCurve",
    "SensitivityResult",
    "TippingPoint",
    "PARAM_DEFS",
    # scorecard
    "SafetyScorecard",
    "ScorecardConfig",
    "ScorecardResult",
    "DimensionScore",
    # optimizer
    "ContractOptimizer",
    "OptimizerConfig",
    "OptimizerResult",
    "CandidateResult",
    "Objective",
    # worker
    "Worker",
    "WorkerState",
    # quarantine
    "QuarantineManager",
    "QuarantineEntry",
    "QuarantineReport",
    "QuarantineStatus",
    "QuarantineSeverity",
    # templates
    "ContractTemplate",
    "TEMPLATES",
    "get_template",
    "list_templates",
    "get_categories",
    "render_catalog",
    "render_comparison_table",
    # incident
    "IncidentResponder",
    "IncidentConfig",
    "IncidentCategory",
    "IncidentSeverity",
    "Playbook",
    "ResponseStep",
    "StepPriority",
    "StepStatus",
    # topology
    "TopologyAnalyzer",
    "TopologyReport",
    "NodeMetrics",
    "SubtreeRisk",
    "RiskLevel",
    "PathologicalPattern",
    # lineage
    "LineageTracker",
    "LineageReport",
    "LineageNode",
    "LineageChain",
    "LineageAnomaly",
    "LineageSeverity",
    "StateMutation",
    "GenerationStats",
    # exporter
    "AuditExporter",
    "ExportConfig",
    "ExportResult",
    # consensus
    "ConsensusProtocol",
    "ConsensusConfig",
    "Proposal",
    "ProposalAction",
    "ProposalStatus",
    "TallyResult",
    "Vote",
    "VoteValue",
    "Decision",
    "ConsensusAuditEntry",
    # game_theory
    "GameTheoryAnalyzer",
    "GameConfig",
    "GameReport",
    "GameType",
    "Move",
    "Payoffs",
    "PairStats",
    "StrategyProfile",
    "StrategyType",
    "StrategicAlert",
    "AlertLevel",
    "Interaction",
    # covert_channels
    "CovertChannelDetector",
    "DetectorConfig",
    "ChannelReport",
    "CovertSignal",
    "AgentMessage",
    "PairProfile",
    "ChannelType",
    # escalation
    "EscalationDetector",
    "EscalationConfig",
    "EscalationResult",
    "EscalationAttempt",
    "EscalationChain",
    "EscalationVector",
    "EscalationSeverity",
    "StealthLevel",
    "AgentEscalationStrategy",
    "AgentPermissions",
    "VectorSummary",
    # killchain
    "KillChainAnalyzer",
    "KillChainConfig",
    "KillChainReport",
    "KillChain",
    "KillChainStage",
    "KillChainAction",
    "KillChainActionCategory",
    "AttackSophistication",
    "ChainStatus",
    "ChainPattern",
    "StageObservation",
    "StageTransition",
    "ACTION_CATALOG",
    "STAGE_ORDER",
    "STAGE_RISK_WEIGHTS",
    "STRATEGY_PROFILES",
    # trust_propagation
    "TrustNetwork",
    "TrustAgent",
    "TrustEdge",
    "TrustReport",
    "ThreatDetection",
    # watermark
    "WatermarkEngine",
    "WatermarkConfig",
    "WatermarkReceipt",
    "WatermarkStrategy",
    "Fingerprint",
    "VerifyResult",
    "VerifyStatus",
    "RobustnessReport",
    "RobustnessResult",
    # threat_correlator
    "ThreatCorrelator",
    "CorrelatorConfig",
    "CorrelationReport",
    "CorrelationRule",
    "CompoundThreat",
    "AgentRisk",
    "CoverageGap",
    "Signal",
    "SignalSource",
    "SignalSeverity",
    "ThreatLevel",
    "ResponseUrgency",
    "CORRELATION_RULES",
    # risk_profiler
    "RiskProfiler",
    "ProfilerConfig",
    "FleetRiskReport",
    "AgentDossier",
    "Finding",
    "FindingSeverity",
    "FindingSource",
    "RiskCategory",
    "RiskTier",
    "CategoryScore",
    "Mitigation",
    # swarm
    "SwarmAnalyzer",
    "SwarmReport",
    "SwarmMetrics",
    "SwarmSignal",
    "SwarmSignalDetection",
    "RoleProfile",
    "SwarmRiskLevel",
    # selfmod
    "SelfModDetector",
    "SelfModConfig",
    "SelfModResult",
    "ModAttempt",
    "ModVector",
    "ModSeverity",
    "ModIntent",
    "AgentAction",
    "ActionType",
    "AgentStrategy",
    "IntentProfile",
    "VectorAnalysis",
    "CorrelationCluster",
    "DetectionRule",
    "BUILTIN_RULES",
    # boundary_tester
    "BoundaryTester",
    "BoundarySpec",
    "BoundaryReport",
    "BoundaryCategory",
    "ProbeResult",
    "ProbeVerdict",
    "FaultInjector",
    # kill_switch
    "KillSwitchManager",
    "TriggerCondition",
    "TriggerKind",
    "KillStrategy",
    "StrategyKind",
    "KillEvent",
    "KillOutcome",
    "EvaluationResult",
    "CooldownEntry",
    "create_conservative_killswitch",
    "create_aggressive_killswitch",
    "create_quarantine_killswitch",
    # anomaly_replay
    "AnomalyReplayer",
    "ReplayConfig",
    "BehaviorTrace",
    "TraceEvent",
    "ReplayControlResult",
    "ReplayReport",
    "ControlVerdict",
    "OverallVerdict",
    "TraceSeverity",
    "TraceLibrary",
    "ReplayCoverageGap",
    "SafetyControl",
    "REPLAY_DEFAULT_CONTROLS",
    # safety_benchmark
    "BenchmarkConfig",
    "BenchmarkGrade",
    "BenchmarkReport",
    "BenchmarkSuite",
    "ComparisonReport",
    "ControlBenchmarkResult",
    "ControlUnderTest",
    "LatencyStats",
    "SafetyBenchmark",
    "WorkloadIntensity",
    # safety_drill
    "DrillConfig",
    "DrillReport",
    "DrillResult",
    "DrillRunner",
    "DrillScenario",
    "DrillVerdict",
    "ReadinessLevel",
    # safety_timeline
    "SafetyTimeline",
    "TimelineEvent",
    "EventSeverity",
    "EventCategory",
    "generate_timeline",
    # trend_tracker
    "TrendTracker",
    "TrendEntry",
    "TrendSummary",
    "RegressionAlert",
    # knowledge_base
    "SafetyKnowledgeBase",
    "KBEntry",
    # alert_router
    "AlertRouter",
    "RoutingRule",
    "Channel",
    "QuietHours",
    "DispatchResult",
    "RouteStats",
    "default_router",
    # persona
    "PersonaSimulator",
    "Persona",
    "PersonaResult",
    "PersonaComparisonResult",
    "PERSONAS",
    # safety_warranty
    "WarrantyManager",
    "Warranty",
    "WarrantyCondition",
    "WarrantyEvaluation",
    "WarrantyReport",
    "WarrantyStatus",
    "WARRANTY_PRESETS",
    # containment_planner
    "ContainmentPlanner",
    "BreachContext",
    "ContainmentPlan",
    "ContainmentStrategy",
    "StrategyRank",
    "ExecutionStep",
    "ContainmentSeverity",
    # runbook
    "RunbookGenerator",
    "ThreatScenario",
    "Runbook",
    "ChecklistItem",
    "EscalationLevel",
    "RecoveryStep",
    "RunbookSeverity",
    "RunbookFormat",
    # blast_radius
    "BlastRadiusAnalyzer",
    "BlastResult",
    "ImpactNode",
    # capability_fingerprint
    "Capability",
    "CapabilityCategory",
    "CapabilityTracker",
    "TrackerConfig",
    "Fingerprint",
    "CapabilityDelta",
    "CapabilityAlert",
    "AlertSeverity",
    # shadow_ai
    "ShadowAIDetector",
    "ScanPolicy",
    "AIInventory",
    "Observation",
    "ShadowAIFinding",
    "ShadowAIReport",
    "FindingCategory",
    "SignalType",
]
