"""Preserve independently captured numerical work and allocation accounting."""

import ast
import hashlib
import json
from math import prod
from pathlib import Path
import subprocess
import sys

import pytest

from support.computations import computation_hashes

ROOT = Path(__file__).resolve().parents[1]
MODULES = json.loads(
    (Path(__file__).parent / "fixtures/kernel-computations.json").read_text()
)["modules"]


@pytest.mark.parametrize("record", MODULES, ids=lambda row: row["module"])
def test_kernel_arithmetic_matches_pre_refactor_source(record):
    observed = set()
    for module in (record["module"], *record.get("related_modules", ())):
        path = ROOT / "src" / Path(module.replace(".", "/")).with_suffix(".py")
        observed.update(computation_hashes(path.read_text()))
    # Additive codecs may introduce new computations. Every frozen computation
    # must still exist, including helpers moved into their format's backend.
    assert set(record["computations"]) <= observed


@pytest.mark.parametrize("precision", ["bfloat16", "fp8"])
def test_state_estimate_matches_allocated_specifications(precision):
    from akas.models.qwen35.config import Qwen35MoEConfig
    from akas.memory.layouts import hybrid_state_specs
    from akas.memory.estimates import state_bytes
    from tensor.runtime.dtypes import dtype

    config = Qwen35MoEConfig(
        layers=("linear_attention", "full_attention"),
        context_limit=4096,
        epsilon=1e-6,
        rope_theta=10000,
    )
    estimate = state_bytes(config, slots=3, context=257, kv_dtype=precision)
    specs = hybrid_state_specs(config, 3, estimate["capacity"], kv_dtype=precision)
    assert sum(estimate[k] for k in ("kv", "recurrent", "convolution")) == sum(
        prod(s.shape) * dtype(s.dtype).itemsize
        for layer in specs.values()
        for s in layer
    )
    if precision == "fp8":
        assert len(specs[1]) == 4  # Both caches and their scales.


def test_runtime_identities_do_not_load_compiler_or_operation_factories():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib.abc, sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.startswith(('tensor.compiler', 'akas.compilation')) or name.split('.')[0] in {'tilelang', 'tvm', 'tvm_ffi', 'torch', 'triton'}:
            raise ImportError(name)
sys.meta_path.insert(0, Guard())
from akas import Engine, LFM2, Qwen35MoE
from akas.artifacts.fingerprint import runtime_hashes, backbone_hashes
for identity in (runtime_hashes, backbone_hashes):
    assert 'akas.ops.attention.cuda' in identity()
assert not any(name.startswith('akas.ops.') and (name.endswith(('.cuda', '.webgpu')) or '.cuda.' in name or '.webgpu.' in name) and not hasattr(module, '__path__') for name, module in sys.modules.items())
""",
        ],
        check=True,
    )


@pytest.mark.parametrize(
    "record",
    json.loads((Path(__file__).parent / "fixtures/kernel-arithmetic.json").read_text())[
        "constants"
    ],
)
def test_inline_hardware_source_matches_pre_refactor_source(record):
    path = ROOT / "src" / Path(record["module"].replace(".", "/")).with_suffix(".py")
    tree = ast.parse(path.read_text())
    node = next(
        n
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == record["name"] for t in n.targets)
    )
    assert (
        hashlib.sha256(ast.literal_eval(node.value).encode()).hexdigest()
        == record["sha256"]
    )
