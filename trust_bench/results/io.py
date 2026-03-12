"""Save and load experiment results as JSON."""

import json
from dataclasses import asdict
from pathlib import Path

from trust_bench.models.base import ProbeResult, ResultMetadata


def save_result(result: ProbeResult, config_path: str) -> str:
    """Save ProbeResult as JSON next to the config file. Returns path."""
    out_dir = Path(config_path).parent
    out_path = out_dir / "results.json"
    with open(out_path, "w") as f:
        json.dump(asdict(result), f, indent=2, default=str)
    return str(out_path)


def load_result(path: str) -> ProbeResult:
    """Load ProbeResult from JSON."""
    with open(path) as f:
        data = json.load(f)
    if isinstance(data.get("result_metadata"), dict):
        data["result_metadata"] = ResultMetadata(**data["result_metadata"])
    return ProbeResult(**data)
