"""Tests for the markdown report generator."""

from pathlib import Path

import pytest

from trust_bench.models.base import ProbeResult, ResultMetadata
from trust_bench.viz.report import generate_report


@pytest.fixture
def feature_survey_result():
    return ProbeResult(
        schema_version="1.0",
        probe_name="feature_survey",
        model_name="llama-3.1-8b",
        config={"probe": "feature_survey", "layer": 16},
        data={
            "total_unique_features": 42,
            "broad_features": [{"feature_idx": 1}, {"feature_idx": 2}],
            "narrow_features": [
                {
                    "feature_idx": 100,
                    "max_activation": 3.5,
                    "categories": ["math", "logic"],
                    "token_count": 7,
                },
                {
                    "feature_idx": 200,
                    "max_activation": 2.1,
                    "categories": ["language"],
                    "token_count": 3,
                },
            ],
        },
        result_metadata=ResultMetadata(
            timestamp="2026-01-01T00:00:00Z",
            trust_bench_version="0.1.0",
            git_hash="abc123",
            device="cpu",
            duration_seconds=1.0,
            layer=16,
            total_tokens=50,
            n_prompts=10,
        ),
    )


def test_generate_report_creates_file(feature_survey_result, tmp_path):
    config_path = str(tmp_path / "config.yaml")
    out_path = generate_report(feature_survey_result, config_path)

    assert out_path == str(tmp_path / "report.md")
    assert Path(out_path).exists()


def test_report_contains_probe_name(feature_survey_result, tmp_path):
    config_path = str(tmp_path / "config.yaml")
    out_path = generate_report(feature_survey_result, config_path)
    content = Path(out_path).read_text()

    assert "feature_survey" in content or "Feature Survey" in content


def test_report_contains_model_name(feature_survey_result, tmp_path):
    config_path = str(tmp_path / "config.yaml")
    out_path = generate_report(feature_survey_result, config_path)
    content = Path(out_path).read_text()

    assert "llama-3.1-8b" in content


def test_report_contains_feature_data(feature_survey_result, tmp_path):
    config_path = str(tmp_path / "config.yaml")
    out_path = generate_report(feature_survey_result, config_path)
    content = Path(out_path).read_text()

    assert "42" in content
    assert "Feature #100" in content
    assert "Feature #200" in content
    assert "math" in content


def test_report_contains_metadata(feature_survey_result, tmp_path):
    config_path = str(tmp_path / "config.yaml")
    out_path = generate_report(feature_survey_result, config_path)
    content = Path(out_path).read_text()

    assert "16" in content  # layer
    assert "50" in content  # total_tokens
    assert "10" in content  # n_prompts


def test_report_returns_path_string(feature_survey_result, tmp_path):
    config_path = str(tmp_path / "config.yaml")
    out_path = generate_report(feature_survey_result, config_path)

    assert isinstance(out_path, str)


def test_report_generic_probe(tmp_path):
    result = ProbeResult(
        schema_version="1.0",
        probe_name="unknown_probe",
        model_name="gpt-2",
        config={"probe": "unknown_probe"},
        data={"key": "value", "count": 99},
        result_metadata=ResultMetadata(
            timestamp="2026-01-01T00:00:00Z",
            trust_bench_version="0.1.0",
            git_hash="def456",
            device="cpu",
            duration_seconds=0.5,
        ),
    )
    config_path = str(tmp_path / "config.yaml")
    out_path = generate_report(result, config_path)
    content = Path(out_path).read_text()

    assert "gpt-2" in content
    assert "99" in content


def test_report_hallucination_probe(tmp_path):
    result = ProbeResult(
        schema_version="1.0",
        probe_name="hallucination",
        model_name="llama-3.1-8b",
        config={"probe": "hallucination"},
        data={
            "differential_features": {
                "16": [
                    {
                        "feature_idx": 55,
                        "fact_mean": 1.2,
                        "control_mean": 0.3,
                        "difference": 0.9,
                    }
                ]
            }
        },
        result_metadata=ResultMetadata(
            timestamp="2026-01-01T00:00:00Z",
            trust_bench_version="0.1.0",
            git_hash="abc123",
            device="cpu",
            duration_seconds=2.0,
        ),
    )
    config_path = str(tmp_path / "config.yaml")
    out_path = generate_report(result, config_path)
    content = Path(out_path).read_text()

    assert "Layer 16" in content
    assert "#55" in content


def test_report_cross_lingual_probe(tmp_path):
    result = ProbeResult(
        schema_version="1.0",
        probe_name="cross_lingual",
        model_name="llama-3.1-8b",
        config={"probe": "cross_lingual"},
        data={
            "cross_lingual_features": [
                {
                    "feature_idx": 77,
                    "concept": "negation",
                    "languages": ["en", "fr", "de"],
                    "n_languages": 3,
                    "max_activation": 4.0,
                    "fires_on_control": False,
                }
            ]
        },
        result_metadata=ResultMetadata(
            timestamp="2026-01-01T00:00:00Z",
            trust_bench_version="0.1.0",
            git_hash="abc123",
            device="cpu",
            duration_seconds=3.0,
        ),
    )
    config_path = str(tmp_path / "config.yaml")
    out_path = generate_report(result, config_path)
    content = Path(out_path).read_text()

    assert "Feature #77" in content
    assert "negation" in content
    assert "en, fr, de" in content
