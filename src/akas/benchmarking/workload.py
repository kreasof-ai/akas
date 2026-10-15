"""The versioned default workload, available in checkouts and installed wheels."""

import hashlib
from importlib.resources import files
import json

_encoded = (
    files("akas.benchmarking").joinpath("workloads/standard-v1.json").read_bytes()
)
STANDARD = json.loads(_encoded)
STANDARD_SHA256 = hashlib.sha256(_encoded).hexdigest()
