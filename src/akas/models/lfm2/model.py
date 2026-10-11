"""LFM2 compositions: optimized packed execution and portable batched reference."""

from akas.execution.bindings.packed.resources import PackedModel
from akas.models.lfm2.weights import LFM2Checkpoint
from akas.models.blocks import DecoderArchitecture
from akas.models.blocks import DecoderBlock


def architecture(config, checkpoint):
    blocks = []
    for index, kind in enumerate(config.layers):
        prefix = f"blk.{index}."
        blocks.append(
            DecoderBlock(
                index,
                prefix,
                kind,
                prefix + ("shortconv." if kind == "conv" else "attn_"),
                "packed",
                prefix + "ffn_",
                prefix + "attn_norm.weight",
                prefix + "ffn_norm.weight",
            )
        )
    head = (
        "output.weight"
        if "output.weight" in checkpoint.gguf.tensors
        else "token_embd.weight"
    )
    return DecoderArchitecture(
        tuple(blocks), "token_embd.weight", "token_embd_norm.weight", head
    )


class LFM2(PackedModel):
    checkpoint_type = LFM2Checkpoint
    definition = staticmethod(architecture)


def compose(
    config,
    *,
    batch=1,
    capacity=None,
    provider="cuda",
    kv_dtype="float16",
    output=False,
    max_weight_bytes=64 * 1024 * 1024,
):
    """One token per slot, with live lengths and masked persistent state.

    Prefill reuses this graph token by token; no kernel changes with length.
    This deliberate reference baseline favors inspectable math over speed.
    """
    from akas.planning import ForwardPlan
    from akas.models.lfm2.weights import tensor_shapes
    from akas.ops.attention.reference import cache_storage

    capacity = min(4096, config.max_context) if capacity is None else capacity
    if (
        type(batch) is not int
        or batch < 1
        or type(capacity) is not int
        or not 1 <= capacity <= config.max_context
    ):
        raise ValueError("batch/capacity must be positive and within model context")
    if provider not in ("cuda", "webgpu"):
        raise ValueError("reference requires CUDA or WebGPU")
    if kv_dtype not in ("float16", "float32"):
        from akas.ops.kv_quantization import get

        get(kv_dtype)
    plan = ForwardPlan(contract="reference")
    b, c = batch, config.width
    from akas.checkpoints.partition import Weights

    weights = Weights(
        plan, tensor_shapes(config, output=output), max_bytes=max_weight_bytes
    )
    tokens = plan.buffer("tokens", (b,), "int32", role="input")
    positions = plan.buffer("positions", (b,), "int32", role="state")
    active = plan.buffer("active", (b,), "int32", role="input")
    hidden = plan.buffer("hidden", (b * c,), "float32")
    for w, start, rows in weights.matrix("token_embd.weight"):
        plan("embedding", dict(r=b, c=c, v=rows, start=start), tokens, w, hidden)

    def scratch(name, width, *, matrix=False):
        return plan.buffer(name, (b, width) if matrix else (b * width,), "float32")

    def norm(name, x, weight, width=c, rows=b):
        out = scratch(name, c if width == c else rows * width // b)
        plan("rms", dict(r=rows, c=width, eps=config.epsilon), x, weights[weight], out)
        return out

    def project(x, weight, k, o, out, *, matrix=False):
        for w, start, rows in weights.matrix(weight):
            plan(
                "linear",
                dict(
                    r=b,
                    k=k,
                    o=rows,
                    matrix_output=matrix,
                    output_stride=o,
                    output_offset=start,
                ),
                x,
                w,
                out,
            )

    def linear(name, x, weight, k, o, *, matrix=False):
        out = scratch(name, o, matrix=matrix)
        project(x, weight, k, o, out, matrix=matrix)
        return out

    def residual(name, x, y):
        out = scratch(name, c)
        plan("add", dict(r=b, c=c), x, y, out)
        return out

    for i, kind in enumerate(config.layers):
        p = f"blk.{i}."
        normalized = norm(p + "normalized", hidden, p + "attn_norm.weight")
        if kind == "conv":
            projection = linear(
                p + "projection", normalized, p + "shortconv.in_proj.weight", c, 3 * c
            )
            state = plan.buffer(
                p + "conv_state", (b, config.conv - 1, c), "float32", role="state"
            )
            mixed = scratch(p + "mixed", c)
            plan(
                "conv",
                dict(r=b, c=c, length=config.conv),
                projection,
                weights[p + "shortconv.conv.weight"],
                state,
                mixed,
                active,
            )
            mixed = linear(
                p + "operator_output", mixed, p + "shortconv.out_proj.weight", c, c
            )
        else:
            d, h, kh = config.head_dim, config.heads, config.kv_heads
            q = linear(p + "q", normalized, p + "attn_q.weight", c, h * d)
            k = linear(p + "k", normalized, p + "attn_k.weight", c, kh * d)
            v = linear(p + "v", normalized, p + "attn_v.weight", c, kh * d, matrix=True)
            q = norm(p + "qn", q, p + "attn_q_norm.weight", width=d, rows=b * h)
            k = norm(p + "kn", k, p + "attn_k_norm.weight", width=d, rows=b * kh)
            qr, kr = (
                scratch(p + "qr", h * d, matrix=True),
                scratch(p + "kr", kh * d, matrix=True),
            )
            plan("rope", dict(r=b, h=h, d=d, theta=config.theta), q, qr, positions)
            plan("rope", dict(r=b, h=kh, d=d, theta=config.theta), k, kr, positions)
            params = dict(
                r=b,
                h=h,
                kh=kh,
                d=d,
                capacity=capacity,
                kv_dtype=kv_dtype,
                provider=provider,
            )
            storage = cache_storage(params)
            kc, vc = [
                plan.buffer(
                    p + name, storage.values.shape, storage.values.dtype, role="state"
                )
                for name in ("kc", "vc")
            ]
            ks, vs = [
                plan.buffer(
                    p + name, storage.metadata[0].shape, "float32", role="state"
                )
                for name in ("ks", "vs")
            ]
            plan("cache_write", params, kr, v, kc, vc, ks, vs, positions, active)
            mixed = scratch(p + "attended", c)
            plan("attention", params, qr, kc, vc, ks, vs, mixed, positions, active)
            mixed = linear(p + "operator_output", mixed, p + "attn_output.weight", c, c)
        hidden = residual(p + "residual", hidden, mixed)
        normalized = norm(p + "ffn_normalized", hidden, p + "ffn_norm.weight")
        gate = linear(p + "gate", normalized, p + "ffn_gate.weight", c, config.ff)
        up = linear(p + "up", normalized, p + "ffn_up.weight", c, config.ff)
        activated = scratch(p + "activated", config.ff)
        plan("swiglu", dict(r=b, c=config.ff), gate, up, activated)
        mixed = linear(p + "ffn_output", activated, p + "ffn_down.weight", config.ff, c)
        hidden = residual(p + "output", hidden, mixed)
    hidden = norm("normalized_output", hidden, "token_embd_norm.weight")
    head = "output.weight" if output else "token_embd.weight"
    logits = plan.buffer("logits", (b * config.vocab,), "float32", role="output")
    project(hidden, head, c, config.vocab, logits)
    sampled = plan.buffer("sampled", (b,), "int32", role="output")
    plan("argmax", dict(r=b, c=config.vocab), logits, sampled, active)
    plan("next_tokens", dict(r=b), sampled, tokens)
    plan("advance", dict(r=b), positions, active)
    return plan


from akas.execution.reference import ReferenceModel
from akas.models.lfm2.weights import ReferenceCheckpoint


class LFM2Reference(ReferenceModel):
    """Batched CUDA/WebGPU example using the same forward plan and checkpoint contract."""

    checkpoint_type = ReferenceCheckpoint
    definition = staticmethod(compose)
