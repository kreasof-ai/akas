"""SwiGLU semantics, storage policies and one kernel definition."""

from akas.ops.registry import variants
from akas.ops.schedules.packed_selection import paired_ffn, projection_tile

IMPLEMENTATIONS = tuple(
    item
    for provider in ("cuda", "webgpu")
    for contract,defaults in (
        ("packed-baseline", (("flat",True),("input_names",("x","y")))),
        ("packed", (("flat",True),("input_names",("x","y")))),
        ("reference", (("flat",True),("input_names",("x","y")))),
        ("bf16", (("packed",True),("output_dtype","bfloat16"),("round_inputs",True))),
        ("block-fp8", (("output_dtype","bfloat16"),("round_inputs",True))),
    )
    if provider == "cuda" or contract in ("packed-baseline", "packed", "reference")
    for item in variants(("swiglu",), "activation.kernel", contract,
                         provider=provider,
                         defaults=defaults)
)
IMPLEMENTATIONS += variants(
    ("swiglu_experts",), "activation.kernel", "block-fp8",
    defaults=(("expert_rows",True),("output_dtype","bfloat16"),("round_inputs",True)),
)
ROW_INDEPENDENT = ("swiglu",)

def validation(kind, p, signature):
    import numpy as np
    from akas.ops.contracts import ValidationCase, sample_arguments

    values = sample_arguments(signature)
    p = {**p, "r": p.get("slots",p.get("r",1))*p.get("chunk",1) if p.get("packed") else p.get("r",1)}
    expected = {name: value.copy() for name, value in values.items()}
    if len(values) == 2 and "x" in values:
        x = values["x"].reshape(p["r"], 2*p["c"])
        gate, up = np.split(x, 2, axis=1)
    else:
        inputs = [v for name,v in values.items() if name != "out"]
        gate, up = inputs
    if p.get("round_inputs"):
        from tensor.runtime.dtypes import encode_bfloat16, decode_bfloat16
        gate = decode_bfloat16(encode_bfloat16(gate))
        up = decode_bfloat16(encode_bfloat16(up))
    out = gate.astype("float64") / (1 + np.exp(-gate.astype("float64"))) * up
    if next(row["dtype"] for row in signature if row["name"] == "out") == "bfloat16":
        from tensor.runtime.dtypes import encode_bfloat16, decode_bfloat16
        out = decode_bfloat16(encode_bfloat16(out))
    expected["out"][:] = out.reshape(expected["out"].shape)
    return ValidationCase(values, expected)


VALIDATIONS = {name: validation for name in ("swiglu","swiglu_experts")}




def packed_ffn(ctx, block):
    prefix = block.feed_forward_prefix
    if (
        ctx.request.provider == "webgpu" or ctx.request.cuda_optimized
    ) and ctx.rows == 1:
        ctx.emit(
            "add_rms",
            dict(r=ctx.rows, c=ctx.width, eps=ctx.config.epsilon),
            ctx.hidden,
            ctx.buffers["mixed"],
            ctx.other,
            ctx.request.weights[block.post_norm],
            ctx.buffers["normal"],
        )
        ctx.hidden, ctx.other = (ctx.other, ctx.hidden)
    else:
        ctx.emit(
            "add",
            dict(r=ctx.rows, c=ctx.width),
            ctx.hidden,
            ctx.buffers["mixed"],
            ctx.other,
        )
        ctx.hidden, ctx.other = (ctx.other, ctx.hidden)
        if ctx.last_only and block.index == len(ctx.config.layers) - 1:
            ctx.emit(
                "last",
                dict(r=ctx.rows, c=ctx.width),
                ctx.hidden,
                ctx.buffers["last"],
                ctx.control,
            )
            ctx.norm(
                block.post_norm, ctx.buffers["last"], ctx.buffers["last_normal"], rows=1
            )
            gate, up = (
                ctx.request.gguf.tensors[prefix + "" + name + ".weight"]
                for name in ("gate", "up")
            )
            ctx.emit(
                "ffn",
                dict(r=1, k=ctx.width, o=ctx.config.ff, type=gate.type),
                ctx.buffers["last_normal"],
                ctx.request.weights[gate.name],
                ctx.request.weights[up.name],
                ctx.buffers["last_activated"],
            )
            info = ctx.request.gguf.tensors[prefix + "down.weight"]
            o, k = info.shape
            ctx.emit(
                "linear_add",
                dict(r=1, k=k, o=o, type=info.type),
                ctx.buffers["last_activated"],
                ctx.request.weights[info.name],
                ctx.buffers["last"],
                ctx.buffers["last_output"],
            )
            ctx.final_hidden = ctx.buffers["last_output"]
            return True
        ctx.norm(block.post_norm, ctx.hidden, ctx.buffers["normal"])
    gate, up = (
        ctx.request.gguf.tensors[prefix + "" + name + ".weight"]
        for name in ("gate", "up")
    )
    if (ctx.request.provider == "webgpu" or ctx.request.cuda_optimized) and paired_ffn(
        ctx.rows,
        gate,
        up,
        "cuda_optimized" if ctx.request.cuda_optimized else ctx.request.webgpu_profile,
    ):
        p = dict(
            r=ctx.rows,
            k=ctx.width,
            o=ctx.config.ff,
            type=gate.type,
            **projection_tile(ctx.rows, ctx.config.ff),
        )
        ctx.emit(
            "ffn",
            p,
            *ctx.inputs("ffn", p, ctx.buffers["normal"]),
            ctx.request.weights[gate.name],
            ctx.request.weights[up.name],
            ctx.buffers["activated"],
        )
    else:
        ctx.linear(prefix + "gate.weight", ctx.buffers["normal"], ctx.buffers["gate"])
        ctx.linear(prefix + "up.weight", ctx.buffers["normal"], ctx.buffers["up"])
        ctx.emit(
            "swiglu",
            dict(r=ctx.rows, c=ctx.config.ff),
            ctx.buffers["gate"],
            ctx.buffers["up"],
            ctx.buffers["activated"],
        )
    if (
        ctx.request.provider == "webgpu" or ctx.request.cuda_optimized
    ) and ctx.rows == 1:
        info = ctx.request.gguf.tensors[prefix + "down.weight"]
        o, k = info.shape
        ctx.emit(
            "linear_add",
            dict(r=ctx.rows, k=k, o=o, type=info.type),
            ctx.buffers["activated"],
            ctx.request.weights[info.name],
            ctx.hidden,
            ctx.other,
        )
    else:
        ctx.linear(
            prefix + "down.weight", ctx.buffers["activated"], ctx.buffers["mixed"]
        )
        ctx.emit(
            "add",
            dict(r=ctx.rows, c=ctx.width),
            ctx.hidden,
            ctx.buffers["mixed"],
            ctx.other,
        )
    ctx.hidden, ctx.other = (ctx.other, ctx.hidden)
