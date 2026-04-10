"""Tests for the model_card module."""
import json
import pytest
from replication.model_card import (
    ModelCardGenerator,
    ModelCardConfig,
    ModelCard,
    RiskEntry,
    RiskLevel,
    RISK_LIBRARY,
    SafetyEvaluation,
)


@pytest.fixture
def generator():
    return ModelCardGenerator()


class TestModelCardConfig:
    def test_from_dict(self):
        d = {"model_name": "TestModel", "model_version": "2.0", "task": "classification"}
        config = ModelCardConfig.from_dict(d)
        assert config.model_name == "TestModel"
        assert config.model_version == "2.0"

    def test_from_dict_ignores_unknown(self):
        d = {"model_name": "X", "unknown_field": 42}
        config = ModelCardConfig.from_dict(d)
        assert config.model_name == "X"

    def test_to_dict(self):
        config = ModelCardConfig(model_name="A")
        d = config.to_dict()
        assert d["model_name"] == "A"


class TestModelCardGenerator:
    def test_generate_basic(self, generator):
        config = ModelCardConfig(model_name="TestAgent", task="testing")
        card = generator.generate(config)
        assert card.config.model_name == "TestAgent"

    def test_known_risks(self, generator):
        config = ModelCardConfig(
            model_name="Risky",
            risks=["prompt_injection", "data_exfiltration"],
        )
        card = generator.generate(config)
        assert len(card.risks) == 2
        assert card.risks[0].name == "Prompt Injection"

    def test_unknown_risk_defaults_medium(self, generator):
        config = ModelCardConfig(risks=["totally_unknown_risk"])
        card = generator.generate(config)
        assert card.risks[0].level == RiskLevel.MEDIUM

    def test_custom_risks(self, generator):
        config = ModelCardConfig(custom_risks=[{
            "name": "Custom Risk",
            "description": "A custom risk",
            "level": "high",
            "mitigations": ["Do something"],
        }])
        card = generator.generate(config)
        assert len(card.risks) == 1
        assert card.risks[0].level == RiskLevel.HIGH

    def test_safety_evaluations(self, generator):
        config = ModelCardConfig(safety_evaluations=[{
            "benchmark": "SafetyBench",
            "score": 85,
            "max_score": 100,
            "date": "2026-01-01",
        }])
        card = generator.generate(config)
        assert len(card.evaluations) == 1
        assert card.evaluations[0].score == 85

    def test_list_known_risks(self, generator):
        risks = generator.list_known_risks()
        assert "prompt_injection" in risks
        assert len(risks) == len(RISK_LIBRARY)


class TestModelCardOutput:
    def test_to_markdown(self):
        gen = ModelCardGenerator()
        config = ModelCardConfig(
            model_name="MarkdownTest",
            risks=["hallucination"],
            description="A test model",
        )
        card = gen.generate(config)
        md = card.to_markdown()
        assert "# Model Card: MarkdownTest" in md
        assert "Hallucination" in md

    def test_to_json(self):
        gen = ModelCardGenerator()
        config = ModelCardConfig(model_name="JsonTest")
        card = gen.generate(config)
        data = json.loads(card.to_json())
        assert data["model_name"] == "JsonTest"

    def test_to_html(self):
        gen = ModelCardGenerator()
        config = ModelCardConfig(model_name="HtmlTest", risks=["toxicity"])
        card = gen.generate(config)
        h = card.to_html()
        assert "<!DOCTYPE html>" in h
        assert "HtmlTest" in h

    def test_overall_risk_level(self):
        gen = ModelCardGenerator()
        config = ModelCardConfig(risks=["data_exfiltration"])  # critical
        card = gen.generate(config)
        assert card.overall_risk_level == RiskLevel.CRITICAL

    def test_to_dict(self):
        gen = ModelCardGenerator()
        config = ModelCardConfig(model_name="DictTest")
        card = gen.generate(config)
        d = card.to_dict()
        assert "risks" in d
        assert d["model_name"] == "DictTest"
