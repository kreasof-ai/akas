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


class NativeDefaultPolicy(ast.NodeTransformer):
    """Specialize only the new opt-in commit controls to their legacy defaults."""

    def visit_Assign(self, node):
        if any(
            isinstance(target, ast.Name) and target.id == "snapshots"
            for target in node.targets
        ):
            assert (
                ast.unparse(node.value)
                == "chunk if p.get('full_checkpoints', False) else chunk - 1"
            )
            return None
        if any(
            isinstance(target, ast.Name) and target.id == "all_tokens"
            for target in node.targets
        ):
            assert ast.unparse(node.value) == "p.get('all_tokens', False)"
            return None
        return self.generic_visit(node)

    def visit_If(self, node):
        if isinstance(node.test, ast.Name) and node.test.id == "commit":
            assert not node.orelse
            return [self.visit(child) for child in node.body]
        return self.generic_visit(node)

    def visit_Name(self, node):
        if node.id == "snapshots":
            return ast.BinOp(
                left=ast.Name(id="chunk", ctx=ast.Load()),
                op=ast.Sub(),
                right=ast.Constant(1),
            )
        return node

    def visit_IfExp(self, node):
        if isinstance(node.test, ast.Name) and node.test.id == "all_tokens":
            assert ast.unparse(node.body) == "0" and ast.unparse(node.orelse) == "1"
            return node.orelse
        return self.generic_visit(node)

    def visit_BinOp(self, node):
        # The new full-window snapshot mode also restores the final prefix.
        if (
            isinstance(node.op, ast.BitOr)
            and ast.unparse(node.right) == "snapshots == chunk"
        ):
            return self.visit(node.left)
        return self.generic_visit(node)


@pytest.mark.parametrize("record", ARITHMETIC, ids=lambda r: r["unit"])
def test_extracted_arithmetic_matches_pre_migration_baseline(record):
    target = record["target"]
    migrated = target.startswith("src/akas/ops/ffn/")
    if migrated and target.endswith("activation.py"):
        from activation_parity import WithoutExpertLayout

        tree = WithoutExpertLayout().visit(
            ast.parse((ROOT / "src/akas/ops/activation/kernel.py").read_text())
        )
    elif migrated:
        from activation_parity import source

        tree = ast.parse(source(target.removeprefix("src/akas/ops/ffn/")))
        # Removed dispatch wrappers are replaced by explicit parameters of the
        # same kernel, with shape/precision parity and native before/after tests.
        from akas.ops.registry import select

        kind = "swiglu_experts" if "swiglu_experts" in record["selector"] else "swiglu"
        contract = (
            "bf16"
            if "bf16_batch" in target
            else "block-fp8"
            if "decode" in target
            else "packed"
        )
        variant = select(
            kind,
            {"r": 2, "c": 257, "top": 3},
            contract=contract,
            phase="decode" if contract == "block-fp8" else "forward",
            target="sm_89",
        )
        assert variant.module == "activation.kernel"
        if kind == "swiglu_experts":
            assert dict(variant.defaults) == {
                "expert_rows": True,
                "output_dtype": "bfloat16",
                "round_inputs": True,
            }
        else:
            from activation_parity import old_factory

            old = old_factory(target.removeprefix("src/akas/ops/ffn/"))(
                kind, {"r": 2, "c": 257}
            )
            from akas.ops.activation import kernel
            from unittest.mock import patch

            with patch.object(kernel, "swiglu_kernel", lambda p: p):
                observed = variant.make({"r": 2, "c": 257})
            assert observed == old
    elif target == "src/akas/ops/gdn/cuda/journal.py":
        tree = ast.parse((ROOT / "src/akas/ops/gdn/verification.py").read_text())
    elif target == "src/akas/ops/speculation/cuda.py":
        tree = ast.parse((ROOT / "src/akas/ops/control/speculative.py").read_text())
    elif target == "src/akas/ops/kv_quantization/cuda/fp8.py":
        tree = ast.parse((ROOT / "src/akas/quantization/kv/fp8.py").read_text())
    elif target == "src/akas/ops/quantization/cuda/block_fp8.py":
        tree = ast.parse(
            (ROOT / "src/akas/quantization/activations/fp8.py").read_text()
        )
    elif target == "src/akas/ops/quantization/cuda/decode.py":
        tree = ast.parse(
            (ROOT / "src/akas/quantization/weights/block_fp8.py").read_text()
        )
    elif target == "src/akas/ops/quantization/cuda/resident_operands.py":
        tree = ast.parse(
            (ROOT / "src/akas/quantization/weights/operands.py").read_text()
        )
    elif target == "src/akas/ops/quantization/webgpu/storage.py":
        owner = (
            "activations/packing.py"
            if record["selector"] == "quantize_kernel"
            else "weights/storage.py"
        )
        tree = ast.parse((ROOT / "src/akas/quantization" / owner).read_text())
    elif target == "src/akas/ops/quantization/webgpu/packed.py":
        # The old dispatcher only forwarded to the same quantize_kernel. There
        # is no mathematical branch to migrate: canonical definition invokes it.
        tree = ast.parse(
            "def make_kernel(kind,p):\n    if kind in ('quantize_q8','quantize_q16'):\n        return quantize_kernel(kind,p)\n    raise ValueError('unsupported operation: ' + kind)\n"
        )
    elif target in (
        "src/akas/ops/quantization/ggml.py",
        "src/akas/ops/quantization/cuda/ggml.py",
    ):
        tree = ast.parse((ROOT / "src/akas/quantization/weights/ggml.py").read_text())
    elif target.startswith("src/akas/ops/residual/"):
        tree = ast.parse((ROOT / "src/akas/ops/residual/kernel.py").read_text())
    elif target == "src/akas/ops/normalization/kernels.py":
        tree = ast.parse((ROOT / "src/akas/ops/normalization/kernel.py").read_text())
    elif target.startswith("src/akas/ops/normalization/"):
        from normalization_parity import source

        tree = ast.parse(source(target.removeprefix("src/akas/ops/normalization/")))
        if (
            record["kind"] == "definition"
            and record["selector"] == "normalization_kernel"
            and "/cuda/" in target
        ):
            tree = ast.parse(
                (ROOT / "src/akas/ops/normalization/kernel.py").read_text()
            )
        elif record["kind"] == "definition" and record["selector"] in (
            "rms_width",
            "rms_kernel",
        ):
            current = ast.parse(
                (ROOT / "src/akas/ops/normalization/kernel.py").read_text()
            )
            name = (
                "parallel_rms_kernel"
                if record["selector"] == "rms_kernel"
                else "rms_width"
            )
            node = next(n for n in current.body if getattr(n, "name", None) == name)
            node.name = record["selector"]
            tree = ast.Module(body=[node], type_ignores=[])
    else:
        hardware_paths = {
            "src/akas/ops/linear/cuda/hopper.py": "src/akas/ops/linear/hardware/hopper/projection.py",
            "src/akas/ops/attention/cuda/hopper_prefill.py": "src/akas/ops/attention/hardware/hopper/prefill.py",
            "src/akas/ops/attention/cuda/hopper.py": "src/akas/ops/attention/hardware/hopper/attention.py",
            "src/akas/ops/gdn/cuda/hopper_restore.py": "src/akas/ops/gdn/restore.py",
            "src/akas/ops/moe/cuda/hopper.py": "src/akas/ops/moe/hardware/hopper/experts.py",
        }
        tree = ast.parse((ROOT / hardware_paths.get(target, target)).read_text())
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
    if (
        record["target"]
        in {
            "src/akas/ops/attention/cuda/quantized.py",
            "src/akas/ops/attention/cuda/fp8.py",
            "src/akas/ops/attention/cuda/split.py",
            "src/akas/ops/attention/cuda/workspace.py",
            "src/akas/ops/attention/cuda/hopper.py",
            "src/akas/ops/attention/cuda/hopper_prefill.py",
        }
        and record["selector"] != "merge"
    ):
        from kv_extraction import restore

        node = restore(node)
    if (
        target == "src/akas/ops/linear/cuda/block_fp8.py"
        and record["selector"] == "make_kernel"
    ):
        node = copy.deepcopy(node)
        assert (
            isinstance(node.body[0], ast.If)
            and ast.unparse(node.body[0].test) == "kind == 'bf16_linear'"
        )
        node.body = node.body[1:]
    if target in {
        "src/akas/ops/gdn/cuda/prefill.py",
        "src/akas/ops/gdn/cuda/verification.py",
        "src/akas/ops/sequence/cuda/prefill.py",
        "src/akas/ops/gdn/cuda/hopper_restore.py",
    }:
        node = NativeDefaultPolicy().visit(copy.deepcopy(node))
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

    config = Qwen35Config(
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
from akas import Engine, LFM2, Qwen35Batch
from akas.artifacts.fingerprint import runtime_hashes, backbone_hashes
for identity in (runtime_hashes, backbone_hashes):
    assert 'akas.ops.attention.cuda.decode' in identity()
assert not any(name.startswith('akas.ops.') and '.cuda.' in name for name in sys.modules)
""",
        ],
        check=True,
    )
