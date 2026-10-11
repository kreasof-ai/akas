"""Compile chunked prefill artifacts without a consumer inference framework."""

from akas.ops.kv_quantization import get, is_quantized, validate as validate_kv_dtype

from pathlib import Path
from akas.compilation.build import build
from akas.compilation.plan import Bundle
from akas.compilation.plan import Kernel
from akas.compilation.plan import Source
from akas.execution.bindings.block_fp8.geometry import validate
from akas.execution.bindings.block_fp8.geometry import metadata
from akas.artifacts.kernels import identity
from akas.execution.bindings.block_fp8.requirements import requirements


def plan(
    checkpoint,
    *,
    slots=8,
    chunk=128,
    context=48000,
    kv_dtype="bfloat16",
    block_m=16,
    packed_kv=False,
    target="sm_89",
):
    validate_kv_dtype(kv_dtype)
    if packed_kv and (
        not is_quantized(kv_dtype)
        or not getattr(get(kv_dtype).backend("cuda"), "PACKED_DTYPE", None)
    ):
        raise ValueError("packed FP8 KV requires the FP8 cache profile")
    c = checkpoint
    validate(c.config)
    rows = slots * chunk
    kernels = []

    def add(kind, p, module, factory, *args):
        kernels.append(
            Kernel(
                identity(kind, p),
                kind,
                p,
                Source(
                    module,
                    factory,
                    args,
                    (
                        "tensor.compiler.frontend.source",
                        "tensor.compiler.cuda.lowering",
                        "akas.ops.attention.cuda.quantized",
                        "akas.ops.normalization.kernels",
                    ),
                ),
            )
        )

    for kind in ("controls", "gdn_conv", "gdn_scan", "expert_routes", "advance"):
        p = dict(slots=slots, chunk=chunk)
        add(kind, p, "akas.compilation.factories.block_fp8", "prefill_factory", kind, p)
    p = dict(slots=slots, chunk=chunk, width=c.config.width)
    add(
        "last_rows",
        p,
        "akas.compilation.factories.block_fp8",
        "prefill_factory",
        "last_rows",
        p,
    )
    for kind in ("attention", "attention_kv"):
        p = dict(slots=slots, chunk=chunk, capacity=context)
        if is_quantized(kv_dtype):
            p["kv_dtype"] = kv_dtype
        if kind == "attention_kv":
            p.update(eps=c.config.epsilon, theta=c.config.rope_theta)
        schedule = (
            dict(p, packed_loads=True, query_rows=128, key_rows=16)
            if kind == "attention" and packed_kv
            else p
        )
        add(
            kind,
            p,
            "akas.compilation.factories.block_fp8",
            "prefill_factory",
            kind,
            schedule,
        )
    needed = requirements(c.config, rows, context)
    skipped = {
        "gdn_conv",
        "gdn_recurrent",
        "attention_kv",
        "attention_partial",
        "attention_merge",
        "argmax",
        "moe_groups",
    }
    for kind, p in needed.values():
        if kind in skipped or (kind == "bf16_linear" and p["o"] == c.config.vocab):
            continue
        if kind == "fp8_linear":
            add(
                kind,
                p,
                "akas.ops.linear.cuda.block_fp8",
                "make_kernel",
                "fp8_linear_mma_prequantized",
                dict(
                    p,
                    columns=64,
                    block_m=block_m,
                    threads=256 if block_m >= 32 else 128,
                    packed_copy=True,
                    stages=2,
                ),
            )
        elif kind == "fp8_experts":
            add(
                kind,
                p,
                "akas.ops.moe.cuda.prefill",
                "expert_kernel",
                dict(
                    rows=rows,
                    k=p["k"],
                    o=p["o"],
                    routed_input=p["routed_input"],
                    block_m=block_m,
                    threads=256 if block_m >= 32 else 128,
                ),
            )
        elif kind == "bf16_linear":
            add(kind, p, "akas.ops.linear.cuda.block_fp8", "bf16_kernel", p)
        else:
            add(
                kind,
                p,
                "akas.compilation.factories.block_fp8",
                "decode_factory",
                kind,
                p,
            )
    for k, top in (
        (c.config.width, None),
        (c.config.value_heads * c.config.value_dim, None),
        (c.config.expert_width, None),
        (c.config.expert_width, c.config.top_k),
    ):
        p = dict(r=rows, k=k)
        if top:
            p["top"] = top
        add("quantize", p, "akas.ops.quantization.cuda.block_fp8", "quantize_kernel", p)
    manifest = dict(
        schema="tensor.qwen35-prefill.v1",
        slots=slots,
        chunk=chunk,
        context=context,
        kv_dtype=kv_dtype,
        target=target,
        schedule=dict(block_m=block_m, packed_kv=packed_kv),
        geometry=metadata(c.config),
        attention_calls=c.config.layers.count("full_attention"),
    )
    return Bundle(manifest, "prefill.json", kernels)


def produce(checkpoint, out, **options):
    build(plan(checkpoint, **options), out)
    return Path(out)
