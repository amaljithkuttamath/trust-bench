import os
import tempfile

from trust_bench.models.base import ProbeResult, ResultMetadata
from trust_bench.results.io import load_result, save_result


class TestResultsIO:
    def _make_result(self):
        return ProbeResult(
            schema_version="1.0",
            probe_name="feature_survey",
            model_name="llama-3.1-8b",
            config={"probe": "feature_survey", "layer": 16},
            data={"narrow_features": [{"feature_idx": 42, "max_activation": 15.0}]},
            result_metadata=ResultMetadata(
                timestamp="2026-03-12T00:00:00Z",
                trust_bench_version="0.1.0",
                git_hash="abc1234",
                device="cpu",
                duration_seconds=2.5,
            ),
        )

    def test_save_and_load_roundtrip(self):
        result = self._make_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.yaml")
            open(config_path, "w").close()

            save_result(result, config_path)

            results_path = os.path.join(tmpdir, "results.json")
            assert os.path.exists(results_path)

            loaded = load_result(results_path)
            assert loaded.probe_name == "feature_survey"
            assert loaded.data["narrow_features"][0]["feature_idx"] == 42

    def test_save_creates_parent_dir(self):
        result = self._make_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "sub", "config.yaml")
            os.makedirs(os.path.dirname(config_path))
            open(config_path, "w").close()

            save_result(result, config_path)
            assert os.path.exists(os.path.join(tmpdir, "sub", "results.json"))

    def test_roundtrip_preserves_metadata(self):
        result = self._make_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.yaml")
            open(config_path, "w").close()
            save_result(result, config_path)
            loaded = load_result(os.path.join(tmpdir, "results.json"))
            assert loaded.result_metadata.canary == "canary-trust-bench-2026"
            assert loaded.result_metadata.git_hash == "abc1234"
