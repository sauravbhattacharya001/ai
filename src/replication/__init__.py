"""AI Replication Sandbox — public API.

Convenience re-exports so consumers can write::

    from replication import Controller, Worker, ReplicationContract, ...

instead of reaching into submodules.
"""

__version__ = "1.0.0"
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
]
