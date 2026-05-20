"""Finding Triage Advisor - agentic safety-finding intake nurse.

Sits at the **intake stage** before the remediation suite:

* :mod:`finding_triage`           - **should we look at this?** (this module)
* :mod:`remediation_planner`      - **what** to fix
* :mod:`remediation_progress`     - **are we** fixing it
* :mod:`remediation_assignment`   - **who** owns each fix
* :mod:`safety_debt`              - **how much debt** has piled up
* :mod:`remediation_roi`          - **which fixes are worth it** this sprint

Receives a fresh batch of raw :class:`Finding` objects (from
``quick_scan`` / ``scorecard`` / ``policy_linter`` / DLP / drift /
custom scanners) and emits a per-finding :class:`TriageVerdict` plus a
deduped, P0-first :class:`TriagePlaybookItem` list:

* HOTFIX_NOW            - page on-call now
* STANDARD_REMEDIATION  - normal queue
* BATCH_WITH_NEXT_SPRINT - bundle next planning window
* DEFER_BACKLOG         - low value, no urgency
* CLOSE_AS_FALSE_POSITIVE - noisy source, not worth tracking
* DEDUPE_OF_EXISTING    - duplicate of a known open item
* ENRICH_AND_RETRIAGE   - too little info to triage cleanly

CLI usage::

    python -m replication triage --demo
    python -m replication triage --demo --format md --risk cautious
    python -m replication triage --demo --output triage.md

Programmatic::

    from replication.remediation_planner import Finding
    from replication.finding_triage import FindingTriageAdvisor

    advisor = FindingTriageAdvisor(risk_appetite="balanced")
    report = advisor.triage([
        Finding(name="kill-switch", source="scorecard", status="fail",
                score=42.0, summary="kill switch race during shutdown"),
        Finding(name="policy-lint", source="policy_linter", status="warn",
                summary="3 overly broad rules"),
    ])
    print(report.to_text())
    print(report.to_markdown())
    print(report.to_json())
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .remediation_planner import Finding
from ._helpers import jaccard as _jaccard_impl


# ── Constants ──────────────────────────────────────────────────────

VALID_APPETITES: Tuple[str, ...] = ("cautious", "balanced", "aggressive")

_SEVERITY_ORDER: Tuple[str, ...] = (
    "info", "low", "medium", "high", "critical",
)
_SEVERITY_WEIGHT: Dict[str, float] = {
    "info": 5.0,
    "low": 20.0,
    "medium": 45.0,
    "high": 70.0,
    "critical": 90.0,
}

_PRIORITY_SLA_HOURS: Dict[str, int] = {
    "P0": 4,
    "P1": 24,
    "P2": 72,
    "P3": 720,
}

_OWNER_BY_SOURCE: Dict[str, str] = {
    "scorecard": "security_eng",
    "quick_scan": "security_eng",
    "policy_linter": "appsec",
    "compliance": "appsec",
    "dlp_scanner": "appsec",
    "performance": "sre",
    "drift": "platform",
    "regression": "sre",
    "ux": "product",
}

_EXPLOIT_PATTERNS: Tuple[str, ...] = (
    "exploit",
    "rce",
    "auth bypass",
    "credential",
    "kill switch",
    "sandbox escape",
    "privilege escalation",
)

_DEFAULT_SOURCE_TRUST: float = 0.7

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


# ── Data model ────────────────────────────────────────────────────


@dataclass(frozen=True)
class TriageVerdict:
    finding_id: str
    finding_name: str
    source: str
    raw_severity: str
    reconciled_severity: str
    verdict: str
    priority: str
    triage_score: float
    reasons: Tuple[str, ...]
    duplicate_of: Optional[str]
    false_positive_confidence: float
    suggested_enrichment: Tuple[str, ...]
    owner_hint: str
    sla_hours: int


@dataclass(frozen=True)
class TriagePlaybookItem:
    id: str
    priority: str
    label: str
    reason: str
    owner: str
    blast_radius: int
    reversibility: str
    finding_ids: Tuple[str, ...]
    suggested_value: Any = None


@dataclass(frozen=True)
class TriageReport:
    generated_at: str
    verdicts: Tuple[TriageVerdict, ...]
    playbook: Tuple[TriagePlaybookItem, ...]
    insights: Tuple[str, ...]
    grade: str
    portfolio_band: str
    summary: str
    counts: Dict[str, int]

    # ── renderers ──

    def to_text(self) -> str:
        buf = io.StringIO()
        buf.write("FINDING TRIAGE REPORT\n")
        buf.write("=" * 60 + "\n")
        buf.write(f"Generated:  {self.generated_at}\n")
        buf.write(f"Grade:      {self.grade}\n")
        buf.write(f"Band:       {self.portfolio_band}\n")
        buf.write(f"Headline:   {self.summary}\n")
        buf.write("\nCounts by verdict:\n")
        for verdict, count in sorted(self.counts.items()):
            buf.write(f"  {verdict:<28} {count}\n")
        buf.write("\nFindings:\n")
        if not self.verdicts:
            buf.write("  (intake quiet — no findings to triage)\n")
        for v in self.verdicts:
            buf.write(
                f"  [{v.priority}] {v.finding_id}  "
                f"{v.verdict}  score={v.triage_score:.1f}  "
                f"sev={v.raw_severity}->{v.reconciled_severity}  "
                f"owner={v.owner_hint}  sla={v.sla_hours}h\n"
            )
            if v.reasons:
                buf.write(f"      reasons: {', '.join(v.reasons)}\n")
            if v.duplicate_of:
                buf.write(f"      duplicate_of: {v.duplicate_of}\n")
            if v.suggested_enrichment:
                buf.write(
                    "      enrich: "
                    + ", ".join(v.suggested_enrichment)
                    + "\n"
                )
        buf.write("\nPlaybook:\n")
        if not self.playbook:
            buf.write("  (none)\n")
        for item in self.playbook:
            buf.write(
                f"  [{item.priority}] {item.id}  {item.label}  "
                f"(owner={item.owner}, blast={item.blast_radius}, "
                f"rev={item.reversibility})\n"
            )
            buf.write(f"      reason: {item.reason}\n")
            if item.finding_ids:
                buf.write(
                    "      covers: " + ", ".join(item.finding_ids) + "\n"
                )
        buf.write("\nInsights:\n")
        if not self.insights:
            buf.write("  (none)\n")
        for ins in self.insights:
            buf.write(f"  - {ins}\n")
        return buf.getvalue()

    def to_markdown(self) -> str:
        buf = io.StringIO()
        buf.write("# Finding Triage Report\n\n")
        buf.write("## Summary\n\n")
        buf.write(f"- Generated: `{self.generated_at}`\n")
        buf.write(f"- Grade: **{self.grade}**\n")
        buf.write(f"- Band: **{self.portfolio_band}**\n")
        buf.write(f"- Headline: {self.summary}\n\n")
        buf.write("### Counts\n\n")
        for verdict, count in sorted(self.counts.items()):
            buf.write(f"- `{verdict}`: {count}\n")
        buf.write("\n## Findings\n\n")
        if not self.verdicts:
            buf.write("_intake quiet — nothing to triage._\n")
        else:
            buf.write(
                "| id | name | source | severity | verdict | "
                "priority | score | reasons |\n"
            )
            buf.write(
                "|----|------|--------|----------|---------|"
                "----------|-------|---------|\n"
            )
            for v in self.verdicts:
                sev = f"{v.raw_severity}→{v.reconciled_severity}"
                reasons = ", ".join(v.reasons) if v.reasons else ""
                buf.write(
                    f"| `{v.finding_id}` | {v.finding_name} | "
                    f"{v.source} | {sev} | {v.verdict} | "
                    f"{v.priority} | {v.triage_score:.1f} | "
                    f"{reasons} |\n"
                )
        buf.write("\n## Playbook\n\n")
        if not self.playbook:
            buf.write("_no playbook items._\n")
        else:
            buf.write(
                "| priority | id | label | owner | "
                "blast | reversibility | covers |\n"
            )
            buf.write(
                "|----------|----|----|-------|------|"
                "----------------|--------|\n"
            )
            for item in self.playbook:
                covers = ", ".join(item.finding_ids) if item.finding_ids else ""
                buf.write(
                    f"| {item.priority} | `{item.id}` | {item.label} | "
                    f"{item.owner} | {item.blast_radius} | "
                    f"{item.reversibility} | {covers} |\n"
                )
        buf.write("\n## Insights\n\n")
        if not self.insights:
            buf.write("_no insights._\n")
        else:
            for ins in self.insights:
                buf.write(f"- {ins}\n")
        return buf.getvalue()

    def to_json(self) -> str:
        return json.dumps(
            asdict(self), sort_keys=True, indent=2, default=str
        )


# ── Helpers ────────────────────────────────────────────────────────


def _slug(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-") or "unknown"


def _finding_id(f: Finding) -> str:
    return f"{_slug(f.source)}:{_slug(f.name)}"


def _tokens(text: str) -> set:
    return set(_TOKEN_RE.findall((text or "").lower()))


def _jaccard(a: set, b: set) -> float:
    return _jaccard_impl(a, b)


def _bump_severity(sev: str, delta: int) -> str:
    if sev not in _SEVERITY_ORDER:
        return sev
    idx = _SEVERITY_ORDER.index(sev)
    idx = max(0, min(len(_SEVERITY_ORDER) - 1, idx + delta))
    return _SEVERITY_ORDER[idx]


def _infer_raw_severity(f: Finding) -> str:
    """Try details.severity first; fall back to status + score heuristics."""
    sev_raw = ""
    details = f.details if isinstance(f.details, dict) else {}
    if details:
        for key in ("severity", "Severity", "level"):
            val = details.get(key)
            if isinstance(val, str) and val.strip():
                sev_raw = val.strip().lower()
                break
    if sev_raw in _SEVERITY_ORDER:
        return sev_raw

    status = (f.status or "").lower()
    score = f.score
    if status == "fail":
        if isinstance(score, (int, float)) and score < 50:
            return "critical"
        return "high"
    if status == "error":
        return "high"
    if status == "warn":
        return "medium"
    if status == "skip":
        return "info"
    if status == "pass":
        return "info"
    return "unknown"


def _has_exploit_hint(text: str) -> bool:
    t = (text or "").lower()
    return any(p in t for p in _EXPLOIT_PATTERNS)


def _owner_for(source: str) -> str:
    return _OWNER_BY_SOURCE.get((source or "").lower(), "unknown")


def _band_from_score(mean_score: float, appetite: str) -> str:
    shift = 0.0
    if appetite == "cautious":
        shift = -5.0
    elif appetite == "aggressive":
        shift = +5.0
    s = mean_score + shift  # cautious shifts bands earlier (lower threshold)
    if s < 20:
        return "CALM"
    if s < 40:
        return "WATCH"
    if s < 60:
        return "ELEVATED"
    if s < 80:
        return "HIGH"
    return "CRITICAL"


# ── Advisor ────────────────────────────────────────────────────────


class FindingTriageAdvisor:
    """Triage raw safety findings into actionable verdicts + playbook."""

    def __init__(
        self,
        *,
        risk_appetite: str = "balanced",
        now: Optional[Callable[[], datetime]] = None,
        false_positive_history: Optional[Dict[str, float]] = None,
        source_trust: Optional[Dict[str, float]] = None,
        existing_findings: Optional[Sequence[Finding]] = None,
        dedupe_jaccard_threshold: float = 0.75,
    ) -> None:
        if risk_appetite not in VALID_APPETITES:
            raise ValueError(
                f"risk_appetite must be one of {VALID_APPETITES}, "
                f"got {risk_appetite!r}"
            )
        if not (0.0 < dedupe_jaccard_threshold <= 1.0):
            raise ValueError(
                "dedupe_jaccard_threshold must be in (0, 1]"
            )
        self.risk_appetite = risk_appetite
        self._now = now or (lambda: datetime.now(timezone.utc))
        self.false_positive_history: Dict[str, float] = dict(
            false_positive_history or {}
        )
        self.source_trust: Dict[str, float] = dict(source_trust or {})
        self.dedupe_jaccard_threshold = float(dedupe_jaccard_threshold)
        self.existing_findings: List[Finding] = list(existing_findings or [])

    # ── public ──

    def triage(self, findings: Sequence[Finding]) -> TriageReport:
        gen_at = self._now().replace(microsecond=0).isoformat()
        findings = list(findings)  # don't mutate caller

        # Pre-tokenize existing findings for dedupe
        existing_tokens: List[Tuple[str, set]] = []
        for ef in self.existing_findings:
            existing_tokens.append((
                _finding_id(ef),
                _tokens(f"{ef.name} {ef.summary}"),
            ))

        verdicts: List[TriageVerdict] = []
        batch_tokens: List[Tuple[str, set]] = []
        for f in findings:
            verdict = self._triage_one(f, existing_tokens, batch_tokens)
            verdicts.append(verdict)
            batch_tokens.append((
                verdict.finding_id,
                _tokens(f"{f.name} {f.summary}"),
            ))

        playbook = self._build_playbook(findings, verdicts)
        insights = self._derive_insights(findings, verdicts)
        counts = self._counts(verdicts)
        grade = self._grade(verdicts, insights)
        mean_score = (
            sum(v.triage_score for v in verdicts) / len(verdicts)
            if verdicts else 0.0
        )
        band = (
            _band_from_score(mean_score, self.risk_appetite)
            if verdicts else "CALM"
        )
        summary = self._headline(verdicts, grade, band)

        return TriageReport(
            generated_at=gen_at,
            verdicts=tuple(verdicts),
            playbook=tuple(playbook),
            insights=tuple(insights),
            grade=grade,
            portfolio_band=band,
            summary=summary,
            counts=counts,
        )

    # ── per-finding triage ──

    def _triage_one(
        self,
        f: Finding,
        existing_tokens: List[Tuple[str, set]],
        batch_tokens: List[Tuple[str, set]],
    ) -> TriageVerdict:
        fid = _finding_id(f)
        raw_sev = _infer_raw_severity(f)
        details = f.details if isinstance(f.details, dict) else {}
        trust = float(self.source_trust.get(f.source, _DEFAULT_SOURCE_TRUST))
        trust = max(0.0, min(1.0, trust))
        fp_rate = float(self.false_positive_history.get(f.source, 0.0))
        fp_rate = max(0.0, min(1.0, fp_rate))

        # Base score from severity (unknown → 25 so we still triage)
        if raw_sev in _SEVERITY_WEIGHT:
            score = _SEVERITY_WEIGHT[raw_sev]
        else:
            score = 25.0

        reasons: List[str] = ["FRESH_INTAKE"]

        # Source trust modifier
        score += (trust * 10.0) - 5.0
        if trust < 0.4:
            reasons.append("LOW_SOURCE_TRUST")

        # False-positive history penalty
        score -= fp_rate * 30.0
        if fp_rate >= 0.3:
            reasons.append("HIGH_FALSE_POSITIVE_RATE")

        # Recency: stale findings get a small boost
        first_seen = details.get("first_seen") if isinstance(details, dict) else None
        if isinstance(first_seen, str):
            try:
                fs = datetime.fromisoformat(first_seen.replace("Z", "+00:00"))
                if fs.tzinfo is None:
                    fs = fs.replace(tzinfo=timezone.utc)
                age_days = (self._now() - fs).total_seconds() / 86400.0
                if age_days >= 30.0:
                    score += 5.0
                    reasons.append("STALE_FINDING")
            except Exception:
                pass

        # Regression bump
        regressed = bool(details.get("regressed")) if isinstance(details, dict) else False
        if regressed:
            score += 10.0
            reasons.append("REGRESSION_DETECTED")

        # Exploitability
        exploit_hit = _has_exploit_hint(f"{f.name} {f.summary}")
        if exploit_hit:
            score += 12.0
            reasons.append("EXPLOITABILITY_HINT")

        # Severity flag
        if raw_sev in ("high", "critical"):
            reasons.append("HIGH_SEVERITY")

        # Enrichment-gap signal (not a score change; flag only)
        summary_text = (f.summary or "").strip()
        thin_summary = len(summary_text) < 30
        no_details = not bool(details)
        unknown_sev = raw_sev == "unknown"
        if thin_summary or no_details or unknown_sev:
            if "INSUFFICIENT_DETAIL" not in reasons:
                reasons.append("INSUFFICIENT_DETAIL")

        # Dedupe matching
        my_tokens = _tokens(f"{f.name} {summary_text}")
        duplicate_of: Optional[str] = None
        best_jac = 0.0
        for canon_id, toks in (existing_tokens + batch_tokens):
            jac = _jaccard(my_tokens, toks)
            if jac >= self.dedupe_jaccard_threshold and jac > best_jac:
                best_jac = jac
                duplicate_of = canon_id
        if duplicate_of is not None:
            score -= 40.0
            reasons.append("DEDUPE_MATCH")

        # Reconciled severity
        reconciled = raw_sev if raw_sev in _SEVERITY_ORDER else "low"
        if "REGRESSION_DETECTED" in reasons or "EXPLOITABILITY_HINT" in reasons:
            reconciled = _bump_severity(reconciled, +1)
        if (
            "HIGH_FALSE_POSITIVE_RATE" in reasons
            and "LOW_SOURCE_TRUST" in reasons
        ):
            reconciled = _bump_severity(reconciled, -1)

        # Risk appetite multiplier
        if self.risk_appetite == "cautious":
            score *= 1.10
        elif self.risk_appetite == "aggressive":
            score *= 0.85

        score = max(0.0, min(100.0, score))

        # False-positive confidence (separate from score)
        fp_conf = max(
            0.0,
            min(
                1.0,
                fp_rate * 0.7 + (1.0 - trust) * 0.3,
            ),
        )

        # Verdict ladder
        verdict = self._pick_verdict(
            raw_sev=raw_sev,
            reconciled=reconciled,
            score=score,
            fp_rate=fp_rate,
            trust=trust,
            duplicate_of=duplicate_of,
            thin_summary=thin_summary,
            no_details=no_details,
            unknown_sev=unknown_sev,
            exploit_hit=exploit_hit,
            regressed=regressed,
        )

        # Priority / SLA per verdict
        priority, sla_hours = self._priority_for(verdict, reconciled)

        # Suggested enrichment
        enrichment: List[str] = []
        if thin_summary:
            enrichment.append("CAPTURE_REPRO_STEPS")
        if no_details:
            enrichment.append("ATTACH_LOG_BUNDLE")
        if unknown_sev:
            enrichment.append("CLARIFY_SEVERITY")
        if fp_rate >= 0.3:
            enrichment.append("CONFIRM_WITH_SECOND_SCANNER")
        if reconciled in ("high", "critical") and not details.get("affected_assets") if isinstance(details, dict) else False:
            enrichment.append("ADD_AFFECTED_ASSET_LIST")
        if "REGRESSION_DETECTED" in reasons:
            enrichment.append("LINK_RELATED_INCIDENT")

        # Stable dedupe of reasons / enrichment while preserving order
        reasons = _dedupe_preserve(reasons)
        enrichment = _dedupe_preserve(enrichment)

        return TriageVerdict(
            finding_id=fid,
            finding_name=f.name,
            source=f.source,
            raw_severity=raw_sev,
            reconciled_severity=reconciled,
            verdict=verdict,
            priority=priority,
            triage_score=round(score, 2),
            reasons=tuple(reasons),
            duplicate_of=duplicate_of,
            false_positive_confidence=round(fp_conf, 3),
            suggested_enrichment=tuple(enrichment),
            owner_hint=_owner_for(f.source),
            sla_hours=sla_hours,
        )

    def _pick_verdict(
        self,
        *,
        raw_sev: str,
        reconciled: str,
        score: float,
        fp_rate: float,
        trust: float,
        duplicate_of: Optional[str],
        thin_summary: bool,
        no_details: bool,
        unknown_sev: bool,
        exploit_hit: bool,
        regressed: bool,
    ) -> str:
        # 1. CLOSE_AS_FALSE_POSITIVE
        if (
            fp_rate >= 0.6
            and trust < 0.5
            and raw_sev in ("info", "low", "medium")
        ):
            return "CLOSE_AS_FALSE_POSITIVE"
        # 2. DEDUPE_OF_EXISTING
        if duplicate_of is not None:
            return "DEDUPE_OF_EXISTING"
        # 3. ENRICH_AND_RETRIAGE
        if thin_summary or no_details or unknown_sev:
            return "ENRICH_AND_RETRIAGE"
        # 4. HOTFIX_NOW
        if reconciled == "critical":
            return "HOTFIX_NOW"
        if score >= 80.0 and exploit_hit:
            return "HOTFIX_NOW"
        if regressed and reconciled in ("high", "critical"):
            return "HOTFIX_NOW"
        # 5/6/7
        if score >= 55.0:
            return "STANDARD_REMEDIATION"
        if score >= 30.0:
            return "BATCH_WITH_NEXT_SPRINT"
        return "DEFER_BACKLOG"

    def _priority_for(
        self, verdict: str, reconciled: str
    ) -> Tuple[str, int]:
        if verdict == "HOTFIX_NOW":
            return "P0", _PRIORITY_SLA_HOURS["P0"]
        if verdict == "STANDARD_REMEDIATION":
            return "P1", _PRIORITY_SLA_HOURS["P1"]
        if verdict == "BATCH_WITH_NEXT_SPRINT":
            return "P2", _PRIORITY_SLA_HOURS["P2"]
        if verdict == "ENRICH_AND_RETRIAGE":
            return "P2", _PRIORITY_SLA_HOURS["P2"]
        if verdict in ("DEDUPE_OF_EXISTING", "CLOSE_AS_FALSE_POSITIVE"):
            return "P3", 168
        return "P3", _PRIORITY_SLA_HOURS["P3"]

    # ── playbook ──

    def _build_playbook(
        self,
        findings: Sequence[Finding],
        verdicts: Sequence[TriageVerdict],
    ) -> List[TriagePlaybookItem]:
        items: List[TriagePlaybookItem] = []

        hotfix_ids = tuple(
            v.finding_id for v in verdicts if v.verdict == "HOTFIX_NOW"
        )
        std_ids = tuple(
            v.finding_id for v in verdicts
            if v.verdict == "STANDARD_REMEDIATION"
        )
        enrich_ids = tuple(
            v.finding_id for v in verdicts
            if v.verdict == "ENRICH_AND_RETRIAGE"
        )
        batch_ids = tuple(
            v.finding_id for v in verdicts
            if v.verdict == "BATCH_WITH_NEXT_SPRINT"
        )
        dupe_ids = tuple(
            v.finding_id for v in verdicts
            if v.verdict == "DEDUPE_OF_EXISTING"
        )
        fp_ids = tuple(
            v.finding_id for v in verdicts
            if v.verdict == "CLOSE_AS_FALSE_POSITIVE"
        )
        regression_ids = tuple(
            v.finding_id for v in verdicts
            if "REGRESSION_DETECTED" in v.reasons
        )

        if hotfix_ids:
            items.append(TriagePlaybookItem(
                id="PAGE_ONCALL_FOR_HOTFIXES",
                priority="P0",
                label="Page on-call for hotfix-class findings",
                reason=(
                    f"{len(hotfix_ids)} HOTFIX_NOW finding(s) require "
                    "immediate response"
                ),
                owner="security_eng",
                blast_radius=4,
                reversibility="low",
                finding_ids=hotfix_ids,
            ))
        if len(hotfix_ids) >= 3:
            items.append(TriagePlaybookItem(
                id="OPEN_INCIDENT_BRIDGE",
                priority="P0",
                label="Open incident bridge for hotfix cluster",
                reason=(
                    f"{len(hotfix_ids)} concurrent HOTFIX_NOW items - "
                    "needs incident coordination"
                ),
                owner="incident_commander",
                blast_radius=5,
                reversibility="low",
                finding_ids=hotfix_ids,
            ))
        if std_ids:
            items.append(TriagePlaybookItem(
                id="ROUTE_STANDARD_QUEUE",
                priority="P1",
                label="Route standard remediations to normal queue",
                reason=f"{len(std_ids)} STANDARD_REMEDIATION item(s)",
                owner="security_eng",
                blast_radius=2,
                reversibility="high",
                finding_ids=std_ids,
            ))

        # Source-noise heuristic for RAISE_SOURCE_TRUST_FLOOR
        source_counts: Dict[str, int] = {}
        source_noise: Dict[str, int] = {}
        for f, v in zip(findings, verdicts):
            source_counts[f.source] = source_counts.get(f.source, 0) + 1
            if v.verdict in (
                "CLOSE_AS_FALSE_POSITIVE", "DEDUPE_OF_EXISTING"
            ):
                source_noise[f.source] = source_noise.get(f.source, 0) + 1

        noisy_source: Optional[str] = None
        for src, count in source_counts.items():
            if count < 2:
                continue
            noise = source_noise.get(src, 0)
            if noise / count >= 0.4 and noise >= 2:
                noisy_source = src
                break

        if len(fp_ids) >= 3 or noisy_source is not None:
            label_src = noisy_source or "noisy_sources"
            items.append(TriagePlaybookItem(
                id="RAISE_SOURCE_TRUST_FLOOR",
                priority="P1",
                label=f"Investigate noisy source: {label_src}",
                reason=(
                    f"{len(fp_ids)} false-positive close(s); "
                    f"noisy source = {label_src}"
                ),
                owner="platform",
                blast_radius=2,
                reversibility="high",
                finding_ids=fp_ids,
                suggested_value=f"review_scanner_{label_src}",
            ))

        if enrich_ids:
            items.append(TriagePlaybookItem(
                id="ENRICH_AND_RETRIAGE",
                priority="P2",
                label="Collect missing detail and re-triage",
                reason=(
                    f"{len(enrich_ids)} finding(s) lacked the detail "
                    "needed to triage cleanly"
                ),
                owner="security_eng",
                blast_radius=1,
                reversibility="high",
                finding_ids=enrich_ids,
            ))
        if batch_ids:
            items.append(TriagePlaybookItem(
                id="BATCH_NEXT_SPRINT",
                priority="P2",
                label="Batch into next remediation sprint",
                reason=f"{len(batch_ids)} BATCH_WITH_NEXT_SPRINT item(s)",
                owner="platform",
                blast_radius=2,
                reversibility="high",
                finding_ids=batch_ids,
            ))
        if len(regression_ids) >= 2:
            items.append(TriagePlaybookItem(
                id="INVESTIGATE_REGRESSION_CLUSTER",
                priority="P2",
                label="Investigate regression cluster",
                reason=(
                    f"{len(regression_ids)} findings flagged as "
                    "regressions - possible upstream change"
                ),
                owner="sre",
                blast_radius=3,
                reversibility="medium",
                finding_ids=regression_ids,
            ))
        if dupe_ids:
            items.append(TriagePlaybookItem(
                id="ARCHIVE_DUPLICATES",
                priority="P3",
                label="Archive duplicate findings to existing canonicals",
                reason=f"{len(dupe_ids)} DEDUPE_OF_EXISTING item(s)",
                owner="platform",
                blast_radius=1,
                reversibility="high",
                finding_ids=dupe_ids,
            ))

        # Aggressive prunes low-priority items when P0/P1 present
        if self.risk_appetite == "aggressive":
            has_p0_p1 = any(i.priority in ("P0", "P1") for i in items)
            if has_p0_p1:
                # drop lone-purpose P3 archive item when P0/P1 burning
                items = [i for i in items if not (
                    i.priority == "P3" and i.id == "ARCHIVE_DUPLICATES"
                    and len(i.finding_ids) <= 1
                )]

        # Fallback when no other actions
        if not items:
            items.append(TriagePlaybookItem(
                id="INTAKE_HEALTHY",
                priority="P3",
                label="Intake healthy - no triage actions required",
                reason="No findings escalated above DEFER_BACKLOG",
                owner="platform",
                blast_radius=1,
                reversibility="high",
                finding_ids=tuple(v.finding_id for v in verdicts),
            ))

        # Cautious appends SCHEDULE_SECOND_OPINION when grade later C/D/F
        # (we don't know grade yet here, so always append in cautious mode
        # when intake non-trivial; main grader will surface it as needed)
        if (
            self.risk_appetite == "cautious"
            and verdicts
            and any(
                v.verdict in (
                    "STANDARD_REMEDIATION",
                    "HOTFIX_NOW",
                    "ENRICH_AND_RETRIAGE",
                ) for v in verdicts
            )
        ):
            items.append(TriagePlaybookItem(
                id="SCHEDULE_SECOND_OPINION",
                priority="P2",
                label="Schedule second-opinion triage review",
                reason="Cautious mode: review borderline calls",
                owner="security_eng",
                blast_radius=1,
                reversibility="high",
                finding_ids=tuple(
                    v.finding_id for v in verdicts
                    if v.verdict in (
                        "STANDARD_REMEDIATION",
                        "ENRICH_AND_RETRIAGE",
                    )
                ),
            ))

        # P0-first ordering, then by id for determinism
        order = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
        items.sort(key=lambda i: (order.get(i.priority, 9), i.id))
        return items

    # ── insights / grade / headline ──

    def _derive_insights(
        self,
        findings: Sequence[Finding],
        verdicts: Sequence[TriageVerdict],
    ) -> List[str]:
        insights: List[str] = []
        n = len(verdicts)
        if n == 0:
            return insights
        if n >= 10:
            insights.append(f"INTAKE_BURST: {n} findings in this batch")
        hotfix = sum(1 for v in verdicts if v.verdict == "HOTFIX_NOW")
        if hotfix >= 2:
            insights.append(f"HOT_INTAKE: {hotfix} hotfix-class findings")
        # noisy source
        source_counts: Dict[str, int] = {}
        source_noise: Dict[str, int] = {}
        for f, v in zip(findings, verdicts):
            source_counts[f.source] = source_counts.get(f.source, 0) + 1
            if v.verdict in (
                "CLOSE_AS_FALSE_POSITIVE", "DEDUPE_OF_EXISTING"
            ):
                source_noise[f.source] = source_noise.get(f.source, 0) + 1
        for src, count in source_counts.items():
            if count / n >= 0.5 and count >= 2:
                noise = source_noise.get(src, 0)
                if noise / count >= 0.4:
                    insights.append(
                        f"NOISY_SOURCE: {src} produced {count}/{n} "
                        f"findings, {noise} noise-class"
                    )
                    break
        regression = sum(
            1 for v in verdicts if "REGRESSION_DETECTED" in v.reasons
        )
        if regression >= 2:
            insights.append(
                f"REGRESSION_CLUSTER: {regression} regressed findings"
            )
        enrich = sum(
            1 for v in verdicts if v.verdict == "ENRICH_AND_RETRIAGE"
        )
        if enrich / n >= 0.3:
            insights.append(
                f"ENRICHMENT_DEBT: {enrich}/{n} need more detail"
            )
        has_p0_p1 = any(v.priority in ("P0", "P1") for v in verdicts)
        if not has_p0_p1:
            insights.append("CLEAN_INTAKE: no P0/P1 escalations")
        return insights

    def _grade(
        self,
        verdicts: Sequence[TriageVerdict],
        insights: Sequence[str],
    ) -> str:
        if not verdicts:
            return "A"
        hotfix = sum(1 for v in verdicts if v.verdict == "HOTFIX_NOW")
        std = sum(1 for v in verdicts if v.verdict == "STANDARD_REMEDIATION")
        has_existing_context = bool(self.existing_findings)
        if hotfix >= 3:
            return "F"
        if hotfix >= 1 and not has_existing_context:
            return "F"
        if hotfix >= 1:
            return "D"
        if any(ins.startswith("ENRICHMENT_DEBT") for ins in insights):
            return "C"
        if std >= 2:
            return "C"
        # remaining: B/A
        if all(
            v.verdict in (
                "CLOSE_AS_FALSE_POSITIVE",
                "DEDUPE_OF_EXISTING",
                "DEFER_BACKLOG",
            ) for v in verdicts
        ):
            return "A"
        return "B"

    def _counts(
        self, verdicts: Sequence[TriageVerdict]
    ) -> Dict[str, int]:
        bins = (
            "HOTFIX_NOW",
            "STANDARD_REMEDIATION",
            "BATCH_WITH_NEXT_SPRINT",
            "DEFER_BACKLOG",
            "CLOSE_AS_FALSE_POSITIVE",
            "DEDUPE_OF_EXISTING",
            "ENRICH_AND_RETRIAGE",
        )
        out = {b: 0 for b in bins}
        for v in verdicts:
            out[v.verdict] = out.get(v.verdict, 0) + 1
        return out

    def _headline(
        self,
        verdicts: Sequence[TriageVerdict],
        grade: str,
        band: str,
    ) -> str:
        if not verdicts:
            return "intake quiet - no findings to triage"
        hotfix = sum(1 for v in verdicts if v.verdict == "HOTFIX_NOW")
        n = len(verdicts)
        if hotfix:
            return (
                f"{hotfix}/{n} hotfix-class - grade {grade}, "
                f"band {band}"
            )
        std = sum(1 for v in verdicts if v.verdict == "STANDARD_REMEDIATION")
        if std:
            return (
                f"{std}/{n} need standard remediation - "
                f"grade {grade}, band {band}"
            )
        return f"{n} findings triaged calmly - grade {grade}, band {band}"


def _dedupe_preserve(items: Sequence[str]) -> List[str]:
    seen: set = set()
    out: List[str] = []
    for x in items:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


# ── Demo + CLI ─────────────────────────────────────────────────────


def _demo_findings() -> List[Finding]:
    """Synthetic intake batch spanning every triage shape."""
    now = datetime.now(timezone.utc)
    stale_iso = (now.replace(microsecond=0)).isoformat()
    # build a stale first_seen by subtracting ~45 days
    from datetime import timedelta
    stale_first_seen = (now - timedelta(days=45)).replace(
        microsecond=0
    ).isoformat()

    return [
        Finding(
            name="kill-switch-race",
            source="scorecard",
            status="fail",
            score=38.0,
            summary=(
                "kill switch race condition during shutdown allows "
                "credential exposure on coordinated restart"
            ),
            details={"severity": "critical", "affected_assets": ["worker-A"]},
        ),
        Finding(
            name="policy-lint-broad-scope",
            source="policy_linter",
            status="warn",
            summary="3 rules with overly broad scope across deploy stages",
            details={"severity": "medium"},
        ),
        Finding(
            name="stale-quick-scan",
            source="quick_scan",
            status="warn",
            summary=(
                "scanner flagged drift in canary thresholds; been open "
                "for a while without retest"
            ),
            details={
                "severity": "medium",
                "first_seen": stale_first_seen,
            },
        ),
        Finding(
            name="duplicate-killish",
            source="scorecard",
            status="fail",
            score=39.0,
            summary=(
                "kill switch race condition during shutdown allows "
                "credential exposure on coordinated restart"
            ),
            details={"severity": "critical"},
        ),
        Finding(
            name="regressed-sla",
            source="regression",
            status="fail",
            score=55.0,
            summary="SLA monitor regressed after release 4.2.1 rollout",
            details={"severity": "high", "regressed": True},
        ),
        Finding(
            name="short",
            source="custom",
            status="warn",
            summary="oops",
            details={},
        ),
    ]


def _ensure_utf8() -> None:
    if sys.stdout.encoding != "utf-8":
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace"
        )


def main(argv: Optional[Sequence[str]] = None) -> int:
    _ensure_utf8()
    parser = argparse.ArgumentParser(
        description=(
            "Agentic finding triage advisor - intake-stage triage of raw "
            "safety findings into HOTFIX_NOW / STANDARD_REMEDIATION / "
            "BATCH / DEFER / CLOSE_FP / DEDUPE / ENRICH verdicts."
        )
    )
    parser.add_argument("--demo", action="store_true",
                        help="Use a built-in synthetic finding batch.")
    parser.add_argument("--format", default="text",
                        choices=("text", "md", "json"))
    parser.add_argument("--risk", default="balanced",
                        choices=VALID_APPETITES)
    parser.add_argument("--output", default=None,
                        help="Write report to this path instead of stdout.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if not args.demo:
        parser.error("only --demo is supported in this CLI")

    findings = _demo_findings()
    advisor = FindingTriageAdvisor(
        risk_appetite=args.risk,
        false_positive_history={"custom": 0.5},
        source_trust={"custom": 0.3, "scorecard": 0.9},
    )
    report = advisor.triage(findings)

    if args.format == "json":
        out = report.to_json()
    elif args.format == "md":
        out = report.to_markdown()
    else:
        out = report.to_text()

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(out)
    else:
        print(out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
