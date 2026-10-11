"""Add routed and dense small-chunk FP8 Split-K projection candidates."""

from akas.compilation.build import build
from akas.compilation.plan import Bundle
from akas.compilation.plan import Kernel
from akas.compilation.plan import Source

import hashlib
import json
from pathlib import Path
import shutil
from akas.artifacts.kernels import identity
from akas.artifacts.fingerprints import (
    fixed_slots_hashes as implementation_hashes,
)


def produce(
    prefill,
    out,
    *,
    expert_parts=4,
    dense_parts=8,
    expert_columns=128,
    grouped_rows=False,
    expert_block_m=16,
):
    prefill, out = Path(prefill).resolve(), Path(out).resolve()
    if out == prefill or out.is_relative_to(prefill):
        raise ValueError("use a separate output")
    manifest = json.loads((prefill / "prefill.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    logicals = list(manifest["kernels"].items())
    records = {}
    for _, row in logicals:
        source = (prefill / row["path"]).resolve()
        if (
            not source.is_relative_to(prefill)
            or hashlib.sha256(source.read_bytes()).hexdigest() != row["sha256"]
        ):
            raise ValueError("source artifact checksum mismatch")
        shutil.copyfile(source, out / row["path"])

    kernels = []

    def add(kind, p, module, factory, *args):
        key = identity(kind, p)
        kernels.append(
            Kernel(
                key,
                kind,
                p,
                Source(module, factory, args),
                target=manifest.get("target", "sm_89"),
            )
        )
        return key

    for logical, row in logicals:
        kind, p = row["kind"], row["parameters"]
        if kind == "bf16_linear" and p["o"] in (1, 32, 256):
            records[logical] = dict(
                matmul=add(
                    "small_bf16",
                    p,
                    "akas.ops.linear.cuda.block_fp8",
                    "bf16_decode_kernel",
                    p,
                )
            )
            continue
        if kind not in ("fp8_experts", "fp8_linear"):
            continue
        grouped = kind == "fp8_experts"
        parts = min(expert_parts if grouped else dense_parts, p["k"] // 128)
        selected = dict(p, partitions=parts)
        if grouped and grouped_rows:
            if p["r"] > 16:
                raise ValueError(
                    "fixed row groups support up to sixteen verification rows"
                )
            matmul = add(
                "split_experts",
                selected,
                "akas.ops.linear.cuda.block_fp8",
                "make_kernel",
                "fp8_experts_mma_prequantized",
                dict(
                    p,
                    columns=expert_columns,
                    block_m=16,
                    partitions=parts,
                    stages=2,
                    packed_copy=True,
                ),
            )
        elif grouped:
            matmul = add(
                "split_experts",
                selected,
                "akas.ops.moe.cuda.split",
                "expert_kernel",
                dict(
                    rows=p["r"],
                    k=p["k"],
                    o=p["o"],
                    routed_input=p["routed_input"],
                    columns=expert_columns,
                    block_m=expert_block_m,
                    threads=256 if expert_block_m >= 32 else 128,
                    partitions=parts,
                    stages=2,
                ),
            )
        else:
            matmul = add(
                "split_dense",
                selected,
                "akas.ops.linear.cuda.block_fp8",
                "make_kernel",
                "fp8_linear_mma_prequantized",
                dict(
                    p,
                    columns=64,
                    block_m=16 if p["r"] <= 16 else (64 if p["r"] >= 64 else 32),
                    threads=256 if p["r"] >= 64 else 128,
                    packed_copy=True,
                    partitions=parts,
                    stages=2,
                ),
            )
        merge_p = dict(r=p["r"], o=p["o"], partitions=parts)
        if grouped:
            merge_p["top"] = 8
        merge = add(
            "split_linear_merge",
            merge_p,
            "akas.ops.linear.cuda.block_fp8",
            "merge_kernel",
            merge_p,
        )
        records[logical] = dict(
            matmul=matmul,
            merge=merge,
            partial_shape=[p["r"], *([8] if grouped else []), parts, p["o"]],
        )
        if grouped and grouped_rows:
            records[logical]["fixed_rows"] = True
    if grouped_rows:
        p = dict(r=manifest["slots"] * manifest["chunk"], top=8)
        key = add(
            "fixed_routes",
            p,
            "akas.profiles.fixed_slots.factories",
            "decode_factory",
            "moe_groups",
            p,
        )
        manifest["fixed_routes"] = key
    manifest.update(
        split_linear=records,
        split_linear_implementation=implementation_hashes(),
        split_linear_schedule=dict(
            expert_parts=expert_parts,
            dense_parts=dense_parts,
            expert_columns=expert_columns,
            grouped_rows=grouped_rows,
            expert_block_m=expert_block_m,
        ),
    )
    build(Bundle(manifest, "prefill.json", kernels), out)
    return out
