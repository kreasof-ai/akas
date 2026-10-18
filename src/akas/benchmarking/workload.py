"""The versioned default workload, available in checkouts and installed wheels."""

import hashlib
from importlib.resources import files
import json
from pathlib import Path


def load(value):
    """Read a packaged workload by ID or a user-supplied JSON path."""
    path = Path(value)
    if path.is_file():
        return json.loads(path.read_text())
    if path.name != str(value) or path.suffix:
        raise ValueError(f"workload file does not exist: {value}")
    resource = files("akas.benchmarking").joinpath(f"workloads/{value}.json")
    if not resource.is_file():
        raise ValueError(f"unknown workload: {value}")
    return json.loads(resource.read_text())


_encoded = (
    files("akas.benchmarking").joinpath("workloads/standard-v1.json").read_bytes()
)
STANDARD = json.loads(_encoded)
STANDARD_SHA256 = hashlib.sha256(_encoded).hexdigest()
