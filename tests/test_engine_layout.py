"""Preserve extracted arithmetic and keep state accounting tied to allocation."""

import ast
import copy
import hashlib
import json
from math import prod
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
ARITHMETIC = json.loads(
    (Path(__file__).parent / "fixtures/engine-arithmetic.json").read_text()
)


class WithoutImports(ast.NodeTransformer):
    def visit_Import(self, node):
        return None

    def visit_ImportFrom(self, node):
        return None


@pytest.mark.parametrize("record", ARITHMETIC, ids=lambda r: r["unit"])
def test_extracted_arithmetic_matches_pre_migration_baseline(record):
    tree = ast.parse((ROOT / record["target"]).read_text())
    if record["kind"] == "definition":
        node = next(
            n for n in tree.body if getattr(n, "name", None) == record["selector"]
        )
    else:
        factory = next(
            n for n in tree.body if getattr(n, "name", None) == "make_kernel"
        )
        node = next(
            n
            for n in factory.body
            if isinstance(n, ast.If) and ast.unparse(n.test) == record["selector"]
        )
    # Module paths may change; arithmetic, signatures, layouts and reduction order may not.
    node = WithoutImports().visit(copy.deepcopy(node))
    observed = hashlib.sha256(
        ast.dump(node, include_attributes=False).encode()
    ).hexdigest()
    assert observed == record["sha256"]


@pytest.mark.parametrize("precision", ["bfloat16", "fp8"])
def test_state_estimate_matches_allocated_specifications(precision):
    from akas.models.qwen35.config import Qwen35Config
    from akas.memory.layouts import hybrid_state_specs
    from akas.memory.estimates import state_bytes
    from tensor.runtime.dtypes import dtype

    config = Qwen35Config(layers=("linear_attention", "full_attention"), context_limit=4096, epsilon=1e-6, rope_theta=10000)
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
from akas import Engine, LFM2, Qwen35Batch
from akas.artifacts.fingerprints import independent_hashes, fixed_slots_hashes, compact_batch_hashes
for identity in (independent_hashes, fixed_slots_hashes, compact_batch_hashes):
    assert 'akas.ops.attention.cuda.decode' in identity()
assert not any(name.startswith('akas.ops.') and '.cuda.' in name for name in sys.modules)
""",
        ],
        check=True,
    )
