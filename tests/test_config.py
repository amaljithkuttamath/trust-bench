import csv
import json
import os
import tempfile

import pytest
import yaml

from trust_bench.config import load_config
from trust_bench.models.base import ConfigError


class TestLoadConfig:
    def _write_config(self, data: dict) -> str:
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
        yaml.dump(data, f)
        f.close()
        return f.name

    def test_loads_valid_config(self):
        path = self._write_config({
            "probe": "feature_survey",
            "model": "llama-3.1-8b",
            "layer": 16,
            "prompts": [],
        })
        config = load_config(path)
        assert config["probe"] == "feature_survey"
        assert config["model"] == "llama-3.1-8b"
        os.unlink(path)

    def test_missing_probe_raises(self):
        path = self._write_config({"model": "llama-3.1-8b"})
        with pytest.raises(ConfigError, match="probe"):
            load_config(path)
        os.unlink(path)

    def test_missing_model_raises(self):
        path = self._write_config({"probe": "feature_survey"})
        with pytest.raises(ConfigError, match="model"):
            load_config(path)
        os.unlink(path)

    def test_nonexistent_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/config.yaml")

    def test_loads_csv_eval_set(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "eval_set.csv")
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["group", "text", "expected"])
                writer.writeheader()
                writer.writerow({
                    "group": "fact", "text": "The capital of France is", "expected": "Paris"
                })
                writer.writerow({
                    "group": "control", "text": "The capital of Freedonia is", "expected": ""
                })

            config_path = os.path.join(tmpdir, "config.yaml")
            with open(config_path, "w") as f:
                yaml.dump({
                    "probe": "hallucination",
                    "model": "llama-3.1-8b",
                    "layers": [16],
                    "eval_set": "eval_set.csv",
                }, f)

            config = load_config(config_path)
            assert "prompts" in config
            assert len(config["prompts"]) == 2
            assert config["prompts"][0]["group"] == "fact"
            assert config["prompts"][0]["text"] == "The capital of France is"

    def test_loads_jsonl_eval_set(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = os.path.join(tmpdir, "eval_set.jsonl")
            with open(jsonl_path, "w") as f:
                f.write(json.dumps({
                    "question": "Is this safe?",
                    "answer_matching_behavior": " Yes",
                    "answer_not_matching_behavior": " No",
                }) + "\n")
                f.write(json.dumps({
                    "question": "Should I help?",
                    "answer_matching_behavior": " Yes",
                    "answer_not_matching_behavior": " No",
                }) + "\n")

            config_path = os.path.join(tmpdir, "config.yaml")
            with open(config_path, "w") as f:
                yaml.dump({
                    "probe": "safety",
                    "model": "llama-3.1-8b",
                    "layers": [16],
                    "eval_set": "eval_set.jsonl",
                }, f)

            config = load_config(config_path)
            assert len(config["prompts"]) == 2
            assert config["prompts"][0]["answer_matching_behavior"] == " Yes"

    def test_csv_eval_set_empty_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "empty.csv")
            with open(csv_path, "w") as f:
                f.write("group,text\n")

            config_path = os.path.join(tmpdir, "config.yaml")
            with open(config_path, "w") as f:
                yaml.dump({
                    "probe": "hallucination",
                    "model": "llama-3.1-8b",
                    "eval_set": "empty.csv",
                }, f)

            with pytest.raises(ConfigError, match="empty"):
                load_config(config_path)
