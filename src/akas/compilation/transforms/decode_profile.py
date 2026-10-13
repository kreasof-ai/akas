"""Produce the native C8 decoder profile selected by complete-state measurements.

The profile remains experimental: kernel correctness and model quality have
separate gates, and complete numerical qualification remains open.
"""

from akas.artifacts.write import publish

import hashlib, json
from pathlib import Path
from akas.compilation.emit import export_source
from akas.compilation.emit import build_artifact
from akas.artifacts.fingerprint import block_fp8_hashes as implementation_hashes
from akas.compilation.recipes.block_fp8.decode import produce as produce_base
from akas.compilation.transforms.projections import mma_bundle


def produce(
    checkpoint, out, *, kv_dtype="bfloat16", context=48000, slots=8, target="sm_89"
):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    base = produce_base(
        checkpoint,
        out / "base",
        slots=slots,
        context=context,
        kv_dtype=kv_dtype,
        target=target,
    )
    decoder = mma_bundle(
        base,
        out / "decoder",
        columns=64,
        partitions=8,
        prequantized=True,
        stages=2,
        packed_copy=True,
    )
    path = decoder / "inference.json"
    manifest = json.loads(path.read_text())

    def compile_row(key, row, module, factory, *args):
        entry = decoder / (key + ".py")
        artifact = decoder / row["path"]
        artifact.unlink()
        entry.write_text(
            export_source(
                module,
                factory,
                *args,
                dependencies=(
                    "tensor.compiler.frontend.source",
                    "tensor.compiler.cuda.lowering",
                    "akas.ops.normalization.kernel",
                ),
            )
        )
        build_artifact(entry, artifact, target=target)
        row["sha256"] = hashlib.sha256(artifact.read_bytes()).hexdigest()

    for key, row in manifest["kernels"].items():
        kind, p = row["kind"], row["parameters"]
        if kind == "bf16_linear":
            if p["o"] in (1, 32, 256):
                compile_row(
                    key, row, "akas.ops.linear.cuda.block_fp8", "bf16_decode_kernel", p
                )
            else:
                compile_row(
                    key,
                    row,
                    "akas.ops.vocabulary.cuda.bf16",
                    "bf16_head_kernel",
                    dict(p, columns=128, depth=128),
                )
        elif kind == "attention_partial" and kv_dtype == "fp8":
            compile_row(
                key,
                row,
                "akas.ops.attention.cuda.quantized",
                "make_kernel",
                kind,
                dict(p, packed_loads=True),
            )
            row["schedule"] = dict(family="packed_fp8_kv")
        elif kind == "fp8_experts":
            parts = min(4, p["k"] // 128)
            schedule = dict(
                columns=128, threads=128, partitions=parts, stages=2, packed_copy=True
            )
            compile_row(
                key,
                row,
                "akas.ops.linear.cuda.block_fp8",
                "make_kernel",
                "fp8_experts_mma_prequantized",
                dict(p, **schedule),
            )
            row["schedule"] = dict(family="mma", **schedule)
            entry = decoder / (key + "-merge.py")
            artifact = entry.with_suffix(".tbin")
            artifact.unlink()
            entry.write_text(
                export_source(
                    "akas.ops.linear.cuda.block_fp8",
                    "merge_kernel",
                    dict(r=slots, top=8, o=p["o"], partitions=parts),
                    dependencies=(
                        "tensor.compiler.frontend.source",
                        "akas.ops.normalization.kernel",
                    ),
                )
            )
            build_artifact(entry, artifact, target=target)
            row["merge"] = dict(
                path=artifact.name,
                sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
            )
    manifest["implementation"] = implementation_hashes()
    manifest["profile_producer_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    manifest.update(model_throughput_qualified=False, full_stress_target_reached=False)
    publish(path, manifest)
    return decoder
