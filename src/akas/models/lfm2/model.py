"""LFM2 compositions using packed or canonical floating checkpoint bindings."""

from functools import partial

from akas.execution.bindings.packed.resources import PackedModel
from akas.execution.model import Model
from akas.models.lfm2.weights import LFM2Checkpoint, ReferenceCheckpoint
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
    chunk=1,
    optimized=False,
):
    """Compose a chunk of tokens per slot with masked, persistent state."""
    from akas.graph import Graph
    from akas.models.lfm2.weights import tensor_shapes
    from akas.ops.attention.reference import cache_storage

    capacity = min(4096, config.max_context) if capacity is None else capacity
    if (
        type(batch) is not int
        or batch < 1
        or type(capacity) is not int
        or not 1 <= capacity <= config.max_context
        or type(chunk) is not int
        or not 1 <= chunk <= capacity
    ):
        raise ValueError("batch/capacity must be positive and within model context")
    if provider not in ("cuda", "webgpu"):
        raise ValueError("reference requires CUDA or WebGPU")
    if kv_dtype not in ("float16", "float32"):
        from akas.quantization.kv import get

        get(kv_dtype)
    plan = Graph(contract="reference")
    b, c = batch * chunk, config.width
    from akas.checkpoints.partition import Weights

    matrix_dtype = "float16" if optimized else "float32"
    weights = Weights(
        plan,
        tensor_shapes(config, output=output),
        max_bytes=max_weight_bytes,
        matrix_dtype=matrix_dtype,
    )
    tokens = plan.buffer("tokens", (batch,), "int32", role="input")
    positions = plan.buffer("positions", (batch,), "int32", role="state")
    active = plan.buffer("active", (batch,), "int32", role="input")
    source_tokens, row_positions, row_active = tokens, positions, active
    if chunk > 1:
        source_tokens = plan.buffer("prefill_tokens", (b,), "int32", role="input")
        row_positions = plan.buffer("row_positions", (b,), "int32")
        row_active = plan.buffer("row_active", (b,), "int32")
        plan(
            "chunk_controls",
            dict(r=batch, chunk=chunk),
            positions,
            active,
            row_positions,
            row_active,
        )
    hidden = plan.buffer("hidden", (b * c,), "float32")
    for w, start, rows in weights.matrix("token_embd.weight"):
        plan(
            "embedding",
            dict(
                r=b,
                c=c,
                v=rows,
                start=start,
                **({"weight_dtype": matrix_dtype} if optimized else {}),
            ),
            source_tokens,
            w,
            hidden,
        )

    def scratch(name, width, *, matrix=False, rows=b):
        return plan.buffer(
            name, (rows, width) if matrix else (rows * width,), "float32"
        )

    def norm(name, x, weight, width=c, rows=b):
        out = scratch(name, width, rows=rows)
        plan("rms", dict(r=rows, c=width, eps=config.epsilon), x, weights[weight], out)
        return out

    def project(x, weight, k, o, out, *, matrix=False, count=b):
        for w, start, rows in weights.matrix(weight):
            plan(
                "linear",
                dict(
                    r=count,
                    k=k,
                    o=rows,
                    matrix_output=matrix,
                    output_stride=o,
                    output_offset=start,
                    **(
                        {"optimized": True, "weight_dtype": matrix_dtype}
                        if optimized
                        else {}
                    ),
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
                p + "conv_state", (batch, config.conv - 1, c), "float32", role="state"
            )
            mixed = scratch(p + "mixed", c)
            plan(
                "conv",
                dict(
                    r=batch,
                    c=c,
                    length=config.conv,
                    **({"chunk": chunk} if chunk > 1 else {}),
                ),
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
            plan("rope", dict(r=b, h=h, d=d, theta=config.theta), q, qr, row_positions)
            plan("rope", dict(r=b, h=kh, d=d, theta=config.theta), k, kr, row_positions)
            params = dict(
                r=b,
                h=h,
                kh=kh,
                d=d,
                capacity=capacity,
                kv_dtype=kv_dtype,
                provider=provider,
                **({"chunk": chunk} if chunk > 1 else {}),
                **({"optimized": True} if optimized else {}),
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
            plan(
                "cache_write", params, kr, v, kc, vc, ks, vs, row_positions, row_active
            )
            mixed = scratch(p + "attended", c)
            from akas.ops.attention.compose import floating_attention

            floating_attention(
                plan, params, qr, kc, vc, ks, vs, mixed, row_positions, row_active
            )
            mixed = linear(p + "operator_output", mixed, p + "attn_output.weight", c, c)
        hidden = residual(p + "residual", hidden, mixed)
        normalized = norm(p + "ffn_normalized", hidden, p + "ffn_norm.weight")
        gate = linear(p + "gate", normalized, p + "ffn_gate.weight", c, config.ff)
        up = linear(p + "up", normalized, p + "ffn_up.weight", c, config.ff)
        activated = scratch(p + "activated", config.ff)
        plan("swiglu", dict(r=b, c=config.ff), gate, up, activated)
        mixed = linear(p + "ffn_output", activated, p + "ffn_down.weight", config.ff, c)
        hidden = residual(p + "output", hidden, mixed)
    if chunk > 1:
        final = scratch("last_hidden", c, rows=batch)
        plan("chunk_last", dict(r=batch, c=c, chunk=chunk), hidden, final, active)
        hidden = final
    hidden = norm("normalized_output", hidden, "token_embd_norm.weight", rows=batch)
    head = "output.weight" if output else "token_embd.weight"
    logits = plan.buffer("logits", (batch * config.vocab,), "float32", role="output")
    projected = (
        scratch("chunk_logits", config.vocab, rows=batch) if chunk > 1 else logits
    )
    project(hidden, head, c, config.vocab, projected, count=batch)
    if chunk > 1:
        plan("masked_copy", dict(r=batch, c=config.vocab), projected, logits, active)
    sampled = plan.buffer("sampled", (batch,), "int32", role="output")
    plan("argmax", dict(r=batch, c=config.vocab), logits, sampled, active)
    plan("next_tokens", dict(r=batch), sampled, tokens)
    plan(
        "advance",
        dict(r=batch, **({"chunk": chunk} if chunk > 1 else {})),
        positions,
        active,
    )
    if optimized:
        from akas.compilation.rewrites import optimize

        return optimize(plan)
    return plan


def compose_program(checkpoint, *, rows, chunk, max_requests, max_context,
                    provider="cuda", kv_dtype="float16",
                    max_weight_bytes=64 * 1024 * 1024):
    """Compose token phases over the shared executor's persistent slot pool."""
    from dataclasses import replace

    graph = compose(checkpoint.config, batch=rows, chunk=chunk,
                    capacity=max_context, provider=provider, kv_dtype=kv_dtype,
                    max_weight_bytes=max_weight_bytes, optimized=True,
                    **checkpoint.definition_options)
    for name, spec in tuple(graph.buffers.items()):
        if spec.role == "state":
            if spec.shape[0] != rows:
                raise ValueError("LFM state must have one leading request dimension")
            value = replace(spec, shape=(max_requests, *spec.shape[1:]))
            object.__setattr__(value, "_graph", graph)
            graph.buffers[name] = value
    graph.buffer("mapping", (rows,), "int32", role="input")
    return graph.bind_requests(
        inputs={"tokens": "prefill_tokens" if chunk > 1 else "tokens",
                "mapping": "mapping", "lengths": "active"},
        prediction="sampled",
        reset=[name for name, spec in graph.buffers.items()
               if spec.role == "state" and (name == "positions" or name.endswith("conv_state"))],
    )


class LFM2Reference(Model):
    """Batched CUDA/WebGPU example using the same forward plan and checkpoint contract."""

    checkpoint_type = ReferenceCheckpoint
    definition = staticmethod(compose)


class LFM2Batched(Model):
    """Canonical loaders and masked slots, using shared optimized operations."""

    checkpoint_type = ReferenceCheckpoint
    definition = staticmethod(partial(compose, optimized=True))
    compilation_profile = "floating"
    compose = staticmethod(compose_program)
