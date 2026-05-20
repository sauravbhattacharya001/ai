"""Regression tests for CWE-79 (stored XSS) hardening in HTML report generators.

These tests pin down the behaviour added in the security fix that:

* HTML-escapes caller-supplied strings before interpolating them into the
  HTML reports produced by ``nutrition_label``, ``fatigue_detector``, and
  ``capability_elicitation``.
* Hardens the JS payload in the ``capability_elicitation`` dashboard so
  that session fields injected via ``innerHTML`` cannot execute script,
  and so JSON blobs cannot break out of the ``<script>`` block.

A regression here would re-introduce a real attack vector: any caller
that ingests untrusted alert metadata, agent names, or session probe
data and renders the resulting report through a static-file viewer
would be vulnerable to stored XSS.
"""

from __future__ import annotations

import pytest

from replication.capability_elicitation import (
    DetectionSignal,
    FleetReport,
    SessionReport,
    ThreatLevel,
)
from replication._helpers import Severity
from replication.fatigue_detector import (
    FatigueIndicator,
    FatigueResult,
    generate_html_report,
)
from replication.nutrition_label import (
    AgentProfile,
    NutritionLabelGenerator,
)


# Common XSS payloads we want to see escaped, not executed.
_XSS_PAYLOAD = "<script>alert('xss')</script>"
_XSS_ATTR = "\" onmouseover=alert(1) x=\""
_ESCAPED_LT = "&lt;script&gt;"


# ── nutrition_label ──────────────────────────────────────────────────


class TestNutritionLabelXssHardening:
    def _label(self, **overrides) -> str:
        profile = AgentProfile(
            agent_name=overrides.get("agent_name", "AgentX"),
            version=overrides.get("version", "1.0"),
            capabilities=overrides.get(
                "capabilities", ["code_execution", "web_access"]
            ),
            safeguards=overrides.get("safeguards", ["sandboxing"]),
            known_hazards=overrides.get(
                "known_hazards", ["prompt_injection"]
            ),
        )
        label = NutritionLabelGenerator().generate(profile)
        # Stuff payloads directly into the label fields so we exercise
        # the f-string interpolations rather than the generator path.
        for key, value in overrides.items():
            if hasattr(label, key):
                setattr(label, key, value)
        return label.to_html()

    def test_agent_name_is_escaped(self):
        html = self._label(agent_name=_XSS_PAYLOAD)
        assert _XSS_PAYLOAD not in html
        assert _ESCAPED_LT in html

    def test_version_is_escaped(self):
        html = self._label(version=_XSS_PAYLOAD)
        assert _XSS_PAYLOAD not in html

    def test_serving_size_is_escaped(self):
        html = self._label(serving_size=_XSS_PAYLOAD)
        assert _XSS_PAYLOAD not in html

    def test_safety_grade_is_escaped(self):
        # Grade is normally "A+"/"B" etc.; a malicious upstream could
        # still inject. Defense in depth.
        html = self._label(safety_grade=_XSS_PAYLOAD)
        assert _XSS_PAYLOAD not in html

    def test_allergen_strings_are_escaped(self):
        html = self._label(allergens=[_XSS_PAYLOAD, "ok"])
        assert _XSS_PAYLOAD not in html
        assert _ESCAPED_LT in html
        assert "ok" in html

    def test_warning_strings_are_escaped(self):
        html = self._label(warnings=[_XSS_PAYLOAD])
        assert _XSS_PAYLOAD not in html
        # Should still render inside an <li>
        assert "<li>" in html

    def test_nutrient_name_and_unit_are_escaped(self):
        # Mutate a real nutrient to carry payload
        profile = AgentProfile(
            agent_name="A",
            capabilities=["code_execution"],
            safeguards=["sandboxing"],
        )
        label = NutritionLabelGenerator().generate(profile)
        assert label.nutrients, "expected generator to emit nutrients"
        label.nutrients[0].name = _XSS_PAYLOAD
        label.nutrients[0].unit = "<img src=x onerror=1>"
        html = label.to_html()
        assert _XSS_PAYLOAD not in html
        assert "<img src=x onerror=1>" not in html
        assert _ESCAPED_LT in html


# ── fatigue_detector ─────────────────────────────────────────────────


class TestFatigueReportXssHardening:
    def _result(self, **overrides) -> FatigueResult:
        ind = FatigueIndicator(
            name=overrides.get("indicator_name", "volume_overload"),
            score=42.0,
            detail=overrides.get("indicator_detail", "ok"),
            severity=overrides.get("indicator_severity", "warning"),
        )
        return FatigueResult(
            score=42.0,
            level=overrides.get("level", "moderate"),
            indicators=[ind],
            recommendations=overrides.get("recommendations", ["ok"]),
            stats={"total_alerts": 1},
        )

    def test_indicator_name_is_escaped(self):
        # ``name`` is passed through ``str.title()`` before rendering,
        # which capitalises ``<script>`` to ``<Script>`` — still an
        # execution vector if interpolated raw, so verify both that
        # the raw angle bracket is escaped and no live tag remains.
        html = generate_html_report(self._result(indicator_name="<svg/onload=1>"))
        assert "<svg/onload=1>" not in html
        assert "&lt;svg/onload=1&gt;" in html or "&lt;Svg/Onload=1&gt;" in html

    def test_indicator_detail_is_escaped(self):
        html = generate_html_report(self._result(indicator_detail=_XSS_PAYLOAD))
        assert _XSS_PAYLOAD not in html
        assert _ESCAPED_LT in html

    def test_indicator_severity_is_escaped(self):
        html = generate_html_report(self._result(indicator_severity=_XSS_PAYLOAD))
        assert _XSS_PAYLOAD not in html

    def test_recommendations_are_escaped(self):
        html = generate_html_report(self._result(recommendations=[_XSS_PAYLOAD]))
        assert _XSS_PAYLOAD not in html
        assert "<li>" in html

    def test_level_string_is_escaped(self):
        # ``result.level`` is normally a fixed enum-ish string, but
        # callers ingesting external scoring data could still pass
        # arbitrary text. Defense in depth: escape it.
        html = generate_html_report(self._result(level=_XSS_PAYLOAD))
        assert _XSS_PAYLOAD not in html


# ── capability_elicitation ───────────────────────────────────────────


def _make_report(
    actor_id: str = "actor-001",
    rec: str = "rotate keys",
    capabilities=None,
    techniques=None,
) -> FleetReport:
    sess = SessionReport(
        actor_id=actor_id,
        threat_level=ThreatLevel.SUSPICIOUS,
        threat_score=42.0,
        signals=[DetectionSignal(engine="DemoEngine", confidence=1.0, severity=Severity.LOW, description="x")],
        probe_count=3,
        duration_seconds=10.0,
        leak_count=0,
        categories_used=["boundary_probe"],
        recommendations=[rec],
    )
    return FleetReport(
        sessions=[sess],
        fleet_threat_score=42.0,
        most_targeted_capabilities=capabilities or [("code_exec", 3)],
        most_common_techniques=techniques or [("jailbreak", 2)],
        active_campaigns=0,
        total_leaks=0,
        recommendations=[rec],
    )


class TestCapabilityElicitationXssHardening:
    def test_recommendations_are_escaped_in_html(self):
        report = _make_report(rec=_XSS_PAYLOAD)
        html = report.to_html()
        # Payload must be HTML-escaped, never echoed raw.
        assert "<li>" + _XSS_PAYLOAD + "</li>" not in html
        assert _ESCAPED_LT in html

    def test_json_payload_cannot_break_out_of_script(self):
        # An actor_id of ``</script><script>alert(1)</script>`` would
        # previously close the inline ``<script>`` block early. After
        # the fix, ``</`` is escaped to ``<\/`` inside JSON literals,
        # so the only ``</script>`` left in the document is the one
        # that legitimately terminates the dashboard's inline script.
        report = _make_report(actor_id="</script><script>alert(1)</script>")
        html = report.to_html()
        assert html.count("</script>") == 1, (
            "JSON-embedded actor_id broke out of <script> block"
        )
        # The escaped form must appear in the JSON instead.
        assert "<\\/script>" in html

    def test_js_runtime_helper_is_present(self):
        # The hardened dashboard must ship a JS-side ``esc()`` helper
        # that wraps every untrusted innerHTML interpolation.
        report = _make_report()
        html = report.to_html()
        assert "function esc(" in html
        # Spot-check that untrusted fields go through it.
        assert "${esc(s.actor_id)}" in html
        assert "${esc(s.threat_level)}" in html
