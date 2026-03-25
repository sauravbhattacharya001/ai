"""Shadow AI Detector — discovers unauthorized AI deployments.

Detects rogue or "shadow" AI systems operating within an organization
that bypass established safety controls, governance policies, and
monitoring pipelines.  Shadow AI is a growing concern: teams may deploy
models, fine-tune APIs, or run inference endpoints without going through
the approved safety review process, creating blind spots in the
organization's AI risk posture.

The detector works by scanning network traffic patterns, API call
signatures, process metadata, and resource usage to identify telltale
signs of AI workloads (GPU utilization spikes, known model-serving
endpoints, embedding API calls, etc.) that are not registered in the
organization's AI inventory.

Usage
-----
::

    from replication.shadow_ai import ShadowAIDetector, ScanPolicy, AIInventory

    inventory = AIInventory(registered_models=["gpt-4", "internal-bert-v2"])
    policy = ScanPolicy(
        scan_network=True,
        scan_processes=True,
        scan_api_calls=True,
    )
    detector = ShadowAIDetector(policy=policy, inventory=inventory)
    report = detector.scan(observations)
    for finding in report.findings:
        print(finding)

    # CLI: python -m replication shadow-ai --observations data.json
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence


class RiskLevel(Enum):
    """Risk level of a shadow AI finding."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class FindingCategory(Enum):
    """Category of shadow AI indicator."""
    UNREGISTERED_MODEL = "unregistered_model"
    ROGUE_ENDPOINT = "rogue_endpoint"
    UNAUTHORIZED_FINE_TUNE = "unauthorized_fine_tune"
    HIDDEN_INFERENCE = "hidden_inference"
    DATA_PIPELINE_LEAK = "data_pipeline_leak"
    SHADOW_EMBEDDING = "shadow_embedding"
    UNLICENSED_API = "unlicensed_api"


class SignalType(Enum):
    """Type of signal that triggered detection."""
    NETWORK_TRAFFIC = "network_traffic"
    PROCESS_METADATA = "process_metadata"
    API_CALL = "api_call"
    GPU_USAGE = "gpu_usage"
    DNS_QUERY = "dns_query"
    LOG_PATTERN = "log_pattern"


# Known AI service domains/patterns for detection
_KNOWN_AI_DOMAINS = [
    r"api\.openai\.com",
    r"api\.anthropic\.com",
    r"generativelanguage\.googleapis\.com",
    r"api\.cohere\.ai",
    r"api-inference\.huggingface\.co",
    r"bedrock-runtime\..*\.amazonaws\.com",
    r".*\.openai\.azure\.com",
    r"api\.replicate\.com",
    r"api\.together\.xyz",
    r"api\.mistral\.ai",
    r"api\.groq\.com",
    r"api\.fireworks\.ai",
    r"api\.perplexity\.ai",
    r"api\.deepseek\.com",
]

_KNOWN_MODEL_PROCESS_PATTERNS = [
    r"vllm",
    r"text-generation-launcher",
    r"tritonserver",
    r"torchserve",
    r"tensorflow_model_server",
    r"ollama",
    r"llama\.cpp",
    r"koboldcpp",
    r"localai",
    r"lm[-_]?studio",
]

_KNOWN_API_KEY_PATTERNS = [
    (r"sk-[a-zA-Z0-9]{20,}", "OpenAI API key"),
    (r"sk-ant-[a-zA-Z0-9\-]{20,}", "Anthropic API key"),
    (r"hf_[a-zA-Z0-9]{20,}", "HuggingFace token"),
    (r"r8_[a-zA-Z0-9]{20,}", "Replicate API key"),
]


@dataclass
class Observation:
    """A single observation from the environment to analyze."""
    signal_type: SignalType
    source: str
    content: str
    timestamp: Optional[str] = None
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class ShadowAIFinding:
    """A detected shadow AI instance."""
    category: FindingCategory
    risk_level: RiskLevel
    signal_type: SignalType
    source: str
    description: str
    evidence: str
    recommendation: str
    matched_pattern: str = ""

    def __str__(self) -> str:
        return (
            f"[{self.risk_level.value.upper()}] {self.category.value}: "
            f"{self.description} (source: {self.source})"
        )


@dataclass
class ShadowAIReport:
    """Full report from a shadow AI scan."""
    findings: List[ShadowAIFinding] = field(default_factory=list)
    observations_scanned: int = 0
    summary: Dict[str, int] = field(default_factory=dict)

    @property
    def has_findings(self) -> bool:
        return len(self.findings) > 0

    @property
    def critical_count(self) -> int:
        return sum(1 for f in self.findings if f.risk_level == RiskLevel.CRITICAL)

    @property
    def high_count(self) -> int:
        return sum(1 for f in self.findings if f.risk_level == RiskLevel.HIGH)

    def by_risk(self, level: RiskLevel) -> List[ShadowAIFinding]:
        return [f for f in self.findings if f.risk_level == level]

    def by_category(self, cat: FindingCategory) -> List[ShadowAIFinding]:
        return [f for f in self.findings if f.category == cat]

    def generate_summary(self) -> Dict[str, int]:
        self.summary = {}
        for level in RiskLevel:
            count = sum(1 for f in self.findings if f.risk_level == level)
            if count:
                self.summary[level.value] = count
        for cat in FindingCategory:
            count = sum(1 for f in self.findings if f.category == cat)
            if count:
                self.summary[cat.value] = count
        self.summary["total"] = len(self.findings)
        return self.summary


@dataclass
class AIInventory:
    """Registry of approved/known AI systems."""
    registered_models: List[str] = field(default_factory=list)
    approved_endpoints: List[str] = field(default_factory=list)
    approved_domains: List[str] = field(default_factory=list)
    approved_api_keys_prefixes: List[str] = field(default_factory=list)

    def is_model_registered(self, model_name: str) -> bool:
        return any(
            m.lower() in model_name.lower()
            for m in self.registered_models
        )

    def is_endpoint_approved(self, endpoint: str) -> bool:
        return any(
            e.lower() in endpoint.lower()
            for e in self.approved_endpoints
        )

    def is_domain_approved(self, domain: str) -> bool:
        return any(
            d.lower() in domain.lower()
            for d in self.approved_domains
        )


@dataclass
class ScanPolicy:
    """Configuration for shadow AI scanning."""
    scan_network: bool = True
    scan_processes: bool = True
    scan_api_calls: bool = True
    scan_gpu: bool = True
    scan_dns: bool = True
    scan_logs: bool = True
    custom_domain_patterns: List[str] = field(default_factory=list)
    custom_process_patterns: List[str] = field(default_factory=list)
    gpu_threshold_percent: float = 50.0


class ShadowAIDetector:
    """Detect unauthorized AI deployments from system observations.

    Parameters
    ----------
    policy : ScanPolicy
        Controls which signal types to scan.
    inventory : AIInventory
        Registry of approved AI systems; anything not in here is shadow.
    """

    def __init__(
        self,
        policy: Optional[ScanPolicy] = None,
        inventory: Optional[AIInventory] = None,
    ) -> None:
        self.policy = policy or ScanPolicy()
        self.inventory = inventory or AIInventory()
        self._domain_patterns = [
            re.compile(p) for p in _KNOWN_AI_DOMAINS
        ] + [re.compile(p) for p in self.policy.custom_domain_patterns]
        self._process_patterns = [
            re.compile(p, re.IGNORECASE) for p in _KNOWN_MODEL_PROCESS_PATTERNS
        ] + [re.compile(p, re.IGNORECASE) for p in self.policy.custom_process_patterns]

    def scan(self, observations: Sequence[Observation]) -> ShadowAIReport:
        """Scan observations for shadow AI indicators."""
        report = ShadowAIReport(observations_scanned=len(observations))

        for obs in observations:
            findings = self._analyze(obs)
            report.findings.extend(findings)

        report.generate_summary()
        return report

    def _analyze(self, obs: Observation) -> List[ShadowAIFinding]:
        """Route an observation to the appropriate analyzer."""
        handlers = {
            SignalType.NETWORK_TRAFFIC: self._check_network,
            SignalType.PROCESS_METADATA: self._check_process,
            SignalType.API_CALL: self._check_api_call,
            SignalType.GPU_USAGE: self._check_gpu,
            SignalType.DNS_QUERY: self._check_dns,
            SignalType.LOG_PATTERN: self._check_logs,
        }
        handler = handlers.get(obs.signal_type)
        if handler is None:
            return []
        return handler(obs)

    def _check_network(self, obs: Observation) -> List[ShadowAIFinding]:
        if not self.policy.scan_network:
            return []
        findings: List[ShadowAIFinding] = []
        for pattern in self._domain_patterns:
            match = pattern.search(obs.content)
            if match and not self.inventory.is_domain_approved(match.group()):
                findings.append(ShadowAIFinding(
                    category=FindingCategory.ROGUE_ENDPOINT,
                    risk_level=RiskLevel.HIGH,
                    signal_type=SignalType.NETWORK_TRAFFIC,
                    source=obs.source,
                    description=f"Unapproved AI API traffic detected: {match.group()}",
                    evidence=obs.content[:200],
                    recommendation="Register this endpoint in the AI inventory or block it.",
                    matched_pattern=pattern.pattern,
                ))
        # Check for API keys in traffic
        for key_pattern, key_name in _KNOWN_API_KEY_PATTERNS:
            if re.search(key_pattern, obs.content):
                findings.append(ShadowAIFinding(
                    category=FindingCategory.UNLICENSED_API,
                    risk_level=RiskLevel.CRITICAL,
                    signal_type=SignalType.NETWORK_TRAFFIC,
                    source=obs.source,
                    description=f"{key_name} detected in network traffic",
                    evidence="[REDACTED — key pattern matched]",
                    recommendation="Rotate this key immediately and investigate the source.",
                    matched_pattern=key_pattern,
                ))
        return findings

    def _check_process(self, obs: Observation) -> List[ShadowAIFinding]:
        if not self.policy.scan_processes:
            return []
        findings: List[ShadowAIFinding] = []
        for pattern in self._process_patterns:
            match = pattern.search(obs.content)
            if match:
                findings.append(ShadowAIFinding(
                    category=FindingCategory.HIDDEN_INFERENCE,
                    risk_level=RiskLevel.HIGH,
                    signal_type=SignalType.PROCESS_METADATA,
                    source=obs.source,
                    description=f"AI model-serving process detected: {match.group()}",
                    evidence=obs.content[:200],
                    recommendation="Verify this is an approved deployment; add to inventory or terminate.",
                    matched_pattern=pattern.pattern,
                ))
        return findings

    def _check_api_call(self, obs: Observation) -> List[ShadowAIFinding]:
        if not self.policy.scan_api_calls:
            return []
        findings: List[ShadowAIFinding] = []

        # Check for model names in API calls
        model_indicators = [
            "gpt-4", "gpt-3.5", "claude", "gemini", "llama",
            "mistral", "command-r", "mixtral", "phi-3", "deepseek",
        ]
        content_lower = obs.content.lower()
        for model in model_indicators:
            if model in content_lower and not self.inventory.is_model_registered(model):
                findings.append(ShadowAIFinding(
                    category=FindingCategory.UNREGISTERED_MODEL,
                    risk_level=RiskLevel.MEDIUM,
                    signal_type=SignalType.API_CALL,
                    source=obs.source,
                    description=f"Unregistered model '{model}' referenced in API call",
                    evidence=obs.content[:200],
                    recommendation=f"Register '{model}' in the AI inventory or remove usage.",
                    matched_pattern=model,
                ))

        # Check for embedding calls (shadow RAG pipelines)
        embedding_patterns = [
            r"/v1/embeddings", r"embed-english", r"text-embedding",
            r"embedding-001", r"/embed\b",
        ]
        for ep in embedding_patterns:
            if re.search(ep, obs.content, re.IGNORECASE):
                findings.append(ShadowAIFinding(
                    category=FindingCategory.SHADOW_EMBEDDING,
                    risk_level=RiskLevel.MEDIUM,
                    signal_type=SignalType.API_CALL,
                    source=obs.source,
                    description="Shadow embedding/RAG pipeline detected",
                    evidence=obs.content[:200],
                    recommendation="Audit this embedding pipeline for data governance compliance.",
                    matched_pattern=ep,
                ))
                break  # one embedding finding per observation is enough

        return findings

    def _check_gpu(self, obs: Observation) -> List[ShadowAIFinding]:
        if not self.policy.scan_gpu:
            return []
        findings: List[ShadowAIFinding] = []
        # Look for GPU utilization numbers in metadata
        gpu_util = obs.metadata.get("gpu_utilization_percent")
        if gpu_util is not None:
            try:
                util_val = float(gpu_util)
            except (ValueError, TypeError):
                return []
            if util_val >= self.policy.gpu_threshold_percent:
                process_name = obs.metadata.get("process_name", "unknown")
                findings.append(ShadowAIFinding(
                    category=FindingCategory.HIDDEN_INFERENCE,
                    risk_level=RiskLevel.MEDIUM,
                    signal_type=SignalType.GPU_USAGE,
                    source=obs.source,
                    description=(
                        f"High GPU utilization ({util_val}%) by '{process_name}' -- "
                        "possible unauthorized model inference"
                    ),
                    evidence=f"GPU at {util_val}%, process: {process_name}",
                    recommendation="Investigate whether this GPU workload is an approved AI deployment.",
                ))
        return findings

    def _check_dns(self, obs: Observation) -> List[ShadowAIFinding]:
        if not self.policy.scan_dns:
            return []
        findings: List[ShadowAIFinding] = []
        for pattern in self._domain_patterns:
            match = pattern.search(obs.content)
            if match and not self.inventory.is_domain_approved(match.group()):
                findings.append(ShadowAIFinding(
                    category=FindingCategory.ROGUE_ENDPOINT,
                    risk_level=RiskLevel.MEDIUM,
                    signal_type=SignalType.DNS_QUERY,
                    source=obs.source,
                    description=f"DNS query to AI service: {match.group()}",
                    evidence=obs.content[:200],
                    recommendation="Investigate who/what is querying this AI endpoint.",
                    matched_pattern=pattern.pattern,
                ))
        return findings

    def _check_logs(self, obs: Observation) -> List[ShadowAIFinding]:
        if not self.policy.scan_logs:
            return []
        findings: List[ShadowAIFinding] = []
        # Fine-tuning indicators
        fine_tune_patterns = [
            r"fine[_-]?tun(e|ing)", r"LoRA", r"QLoRA", r"training[_\s]?loss",
            r"adapter[_\s]?weights", r"checkpoint[_\s]?saved",
        ]
        for fp in fine_tune_patterns:
            if re.search(fp, obs.content, re.IGNORECASE):
                findings.append(ShadowAIFinding(
                    category=FindingCategory.UNAUTHORIZED_FINE_TUNE,
                    risk_level=RiskLevel.CRITICAL,
                    signal_type=SignalType.LOG_PATTERN,
                    source=obs.source,
                    description="Unauthorized model fine-tuning activity detected in logs",
                    evidence=obs.content[:200],
                    recommendation="Halt fine-tuning immediately; review data governance and model approval.",
                    matched_pattern=fp,
                ))
                break  # one fine-tuning finding per observation

        # Data pipeline indicators
        pipeline_patterns = [
            r"vector[_\s]?store", r"pinecone", r"weaviate", r"chroma[_\s]?db",
            r"qdrant", r"milvus", r"faiss[_\s]?index",
        ]
        for pp in pipeline_patterns:
            if re.search(pp, obs.content, re.IGNORECASE):
                findings.append(ShadowAIFinding(
                    category=FindingCategory.DATA_PIPELINE_LEAK,
                    risk_level=RiskLevel.HIGH,
                    signal_type=SignalType.LOG_PATTERN,
                    source=obs.source,
                    description="Unauthorized vector database / RAG pipeline detected",
                    evidence=obs.content[:200],
                    recommendation="Audit data ingestion pipeline for PII and sensitive data.",
                    matched_pattern=pp,
                ))
                break

        return findings


def main(argv: Optional[List[str]] = None) -> None:
    """CLI: ``python -m replication shadow-ai --observations data.json``"""
    import argparse as _ap
    import json as _json
    import sys as _sys

    parser = _ap.ArgumentParser(
        prog="replication shadow-ai",
        description="Detect unauthorized shadow AI deployments",
    )
    parser.add_argument(
        "--observations", "-o",
        help="JSON file with observations (list of objects with signal_type, source, content)",
    )
    parser.add_argument(
        "--models", "-m", nargs="*", default=[],
        help="Registered/approved model names",
    )
    parser.add_argument(
        "--domains", "-d", nargs="*", default=[],
        help="Approved AI domains",
    )
    parser.add_argument(
        "--json", dest="as_json", action="store_true",
        help="Output as JSON",
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Run a built-in demo scan",
    )
    args = parser.parse_args(argv)

    inventory = AIInventory(
        registered_models=args.models,
        approved_domains=args.domains,
    )
    detector = ShadowAIDetector(inventory=inventory)

    if args.demo:
        observations = [
            Observation(SignalType.NETWORK_TRAFFIC, "proxy-01",
                        "POST https://api.openai.com/v1/chat/completions"),
            Observation(SignalType.PROCESS_METADATA, "node-gpu-3",
                        "vllm serve --model meta-llama/Llama-3-70B"),
            Observation(SignalType.API_CALL, "service-A",
                        '{"model": "claude-3-opus", "messages": [...]}'),
            Observation(SignalType.LOG_PATTERN, "ml-team-node",
                        "LoRA fine-tuning complete, checkpoint saved at epoch 5"),
            Observation(SignalType.DNS_QUERY, "dns-sinkhole",
                        "QUERY api.anthropic.com A"),
            Observation(SignalType.GPU_USAGE, "workstation-7",
                        "GPU utilization report",
                        metadata={"gpu_utilization_percent": "92", "process_name": "python"}),
            Observation(SignalType.LOG_PATTERN, "data-eng-box",
                        "Indexing 50k documents into chromadb vector_store"),
        ]
    elif args.observations:
        with open(args.observations, "r", encoding="utf-8") as fh:
            raw = _json.load(fh)
        observations = [
            Observation(
                signal_type=SignalType(item["signal_type"]),
                source=item.get("source", "unknown"),
                content=item.get("content", ""),
                timestamp=item.get("timestamp"),
                metadata=item.get("metadata", {}),
            )
            for item in raw
        ]
    else:
        print("Provide --observations FILE or --demo.  Use --help for details.")
        _sys.exit(1)

    report = detector.scan(observations)

    if args.as_json:
        findings = [
            {
                "category": f.category.value,
                "risk_level": f.risk_level.value,
                "signal_type": f.signal_type.value,
                "source": f.source,
                "description": f.description,
                "recommendation": f.recommendation,
            }
            for f in report.findings
        ]
        print(_json.dumps({
            "observations_scanned": report.observations_scanned,
            "total_findings": len(report.findings),
            "summary": report.summary,
            "findings": findings,
        }, indent=2))
    else:
        print(f"Shadow AI Scan -- {report.observations_scanned} observations analyzed")
        print(f"{'=' * 60}")
        if not report.findings:
            print("[OK] No shadow AI detected.")
        else:
            print(f"[!] {len(report.findings)} finding(s):\n")
            for f in report.findings:
                print(f"  [{f.risk_level.value.upper():>8}] {f.description}")
                print(f"           Source: {f.source}")
                print(f"           -> {f.recommendation}\n")
            print(f"Summary: {report.summary}")
        if report.critical_count:
            _sys.exit(2)
        elif report.high_count:
            _sys.exit(1)
