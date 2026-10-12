"""Compile a model-supplied token-step composition using shared op selection."""


def produce(
    checkpoint,
    out,
    *,
    definition,
    batch=1,
    capacity=4096,
    provider="cuda",
    target=None,
    kv_dtype="float16",
    schedules=None,
    max_weight_bytes=64 * 1024 * 1024,
    prefill_chunk=1,
):
    from akas.compilation.forward import plan
    from akas.compilation.build import build

    forward = definition(
        checkpoint.config,
        batch=batch,
        capacity=capacity,
        provider=provider,
        kv_dtype=kv_dtype,
        output="output.weight" in checkpoint.names,
        max_weight_bytes=max_weight_bytes,
    )
    bundle = plan(forward, provider=provider, target=target, schedules=schedules)
    bundle.manifest["reference"] = dict(
        batch=batch,
        capacity=capacity,
        kv_dtype=kv_dtype,
        max_weight_bytes=max_weight_bytes,
        **({"prefill_chunk": prefill_chunk} if prefill_chunk > 1 else {}),
        **(
            {"matrix_dtype": "float16"}
            if any(
                b.role == "weight" and b.dtype == "float16"
                for b in forward.buffers.values()
            )
            else {}
        ),
    )
    if prefill_chunk > 1:
        prefill = definition(
            checkpoint.config,
            batch=batch,
            capacity=capacity,
            provider=provider,
            kv_dtype=kv_dtype,
            output="output.weight" in checkpoint.names,
            max_weight_bytes=max_weight_bytes,
            chunk=prefill_chunk,
        )
        build(
            plan(
                prefill,
                provider=provider,
                target=target,
                schedules=schedules,
                filename="prefill.json",
            ),
            out,
        )
    return build(bundle, out)
