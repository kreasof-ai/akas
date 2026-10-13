"""Masked batched RoPE and online GQA baseline, independent of model names.

Reference head width is 64. Work and cache addresses use live positions rather
than recompiling for each sequence length. Codecs own all quantization math.
"""

import math


def cache_storage(p):
    from akas.quantization.kv import get
    from akas.quantization.kv.codec import Storage
    from akas.memory.state import BufferSpec

    shape = (p["r"] // p.get("chunk", 1), p["kh"], p["capacity"], p["d"])
    if p["kv_dtype"] not in ("float16", "float32"):
        return get(p["kv_dtype"]).storage(shape, provider=p["provider"])
    dtype = "float32" if p["provider"] == "webgpu" else p["kv_dtype"]
    return Storage(BufferSpec(shape, dtype), (BufferSpec((*shape[:-1], 1), "float32"),))


def make_kernel(kind, p):
    import tilelang.language as T
    from tensor.compiler.frontend.source import primitive
    from akas.ops.reductions.webgpu import tree_reduce

    r, d = p["r"], p["d"]
    if d != 64:
        raise ValueError("portable reference attention requires 64-channel heads")
    if kind == "rope":
        h, theta = p["h"], p["theta"]

        @T.prim_func
        def kernel(
            x: T.Tensor((r * h * d,), "float32"),
            out: T.Tensor((r, h * d), "float32"),
            positions: T.Tensor((r,), "int32"),
        ):
            with T.Kernel(h, r, threads=64) as (head, row):
                tx = T.get_thread_binding()
                half = tx % (d // 2)
                angle = T.cast(positions[row], "float32") * T.exp(
                    -(math.log(theta) * 2 / d) * half
                )
                left = x[(row * h + head) * d + half]
                right = x[(row * h + head) * d + half + d // 2]
                out[row, head * d + tx] = T.if_then_else(
                    tx < d // 2,
                    left * T.cos(angle) - right * T.sin(angle),
                    left * T.sin(angle) + right * T.cos(angle),
                )

        return kernel
    h, kh, cap = p["h"], p["kh"], p["capacity"]
    chunk = p.get("chunk", 1)
    storage = cache_storage(p)
    shape, dtype = storage.values.shape, storage.values.dtype
    if len(storage.metadata) != 1:
        raise ValueError("reference attention requires one metadata buffer per cache")
    scale_shape = storage.metadata[0].shape
    quantized = p["kv_dtype"] not in ("float16", "float32")
    if quantized:
        from akas.quantization.kv import get

        backend = get(p["kv_dtype"]).backend(p["provider"])
        encode = backend.reference_writer(d, chunk=chunk)
        read = backend.reference_decoder()
    else:
        from akas.quantization.weights.storage import round_half

        rounded = round_half()

        @T.macro
        def read(cache, metadata, slot, head, position, channel):
            return (
                T.cast(cache[slot, head, position, channel], "float32")
                * metadata[slot, head, position, 0]
            )

    if kind == "cache_write":

        @T.macro
        def write(keys, values, kc, vc, ks, vs, positions, enabled):
            with T.Kernel(kh, r, threads=128) as (head, row):
                tx = T.get_thread_binding()
                if quantized:
                    encode(
                        keys,
                        values,
                        row,
                        head,
                        positions[row],
                        kc,
                        vc,
                        ks,
                        vs,
                        tx,
                        enabled,
                    )
                else:
                    if enabled[row] != 0:
                        if tx == 0:
                            ks[row // chunk, head, positions[row], 0] = 1
                            vs[row // chunk, head, positions[row], 0] = 1
                        if tx < d:
                            if p["kv_dtype"] == "float16" and p["provider"] == "webgpu":
                                kc[row // chunk, head, positions[row], tx] = rounded(
                                    keys[row, head * d + tx]
                                )
                                vc[row // chunk, head, positions[row], tx] = rounded(
                                    values[row, head * d + tx]
                                )
                            else:
                                kc[row // chunk, head, positions[row], tx] = keys[
                                    row, head * d + tx
                                ]
                                vc[row // chunk, head, positions[row], tx] = values[
                                    row, head * d + tx
                                ]

        return primitive(
            [
                ("keys", (r, kh * d), "float32"),
                ("values", (r, kh * d), "float32"),
                ("kc", shape, dtype),
                ("vc", shape, dtype),
                ("ks", scale_shape, "float32"),
                ("vs", scale_shape, "float32"),
                ("positions", r, "int32"),
                ("enabled", r, "int32"),
            ],
            write,
        )
    if kind != "attention":
        raise ValueError(kind)
    total = tree_reduce(64)

    @T.macro
    def attend(q, kc, vc, ks, vs, out, positions, enabled):
        with T.Kernel(h, r, threads=64) as (head, row):
            tx = T.get_thread_binding()
            scratch = T.alloc_shared((64,), "float32")
            maximum = T.alloc_shared((1,), "float32")
            normalizer = T.alloc_shared((1,), "float32")
            factor = T.alloc_shared((1,), "float32")
            probability = T.alloc_shared((1,), "float32")
            accum = T.alloc_var("float32")
            accum = 0
            if tx == 0:
                maximum[0] = -T.infinity("float32")
                normalizer[0] = 0
            T.sync_threads()
            for position in T.serial(
                cap if p["provider"] == "webgpu" else T.min(positions[row] + 1, cap)
            ):
                scratch[tx] = q[row, head * d + tx] * read(
                    kc, ks, row // chunk, head // (h // kh), position, tx
                )
                T.sync_threads()
                total(scratch, tx)
                if tx == 0:
                    score = T.if_then_else(
                        position <= positions[row],
                        scratch[0] * d ** (-0.5),
                        T.float32(-1e30),
                    )
                    next_max = T.max(maximum[0], score)
                    factor[0] = T.exp(maximum[0] - next_max)
                    probability[0] = T.if_then_else(
                        position <= positions[row], T.exp(score - next_max), 0
                    )
                    normalizer[0] = normalizer[0] * factor[0] + probability[0]
                    maximum[0] = next_max
                T.sync_threads()
                accum = accum * factor[0] + probability[0] * read(
                    vc, vs, row // chunk, head // (h // kh), position, tx
                )
                T.sync_threads()
            if enabled[row] != 0:
                out[row * h * d + head * d + tx] = accum / normalizer[0]
            else:
                out[row * h * d + head * d + tx] = 0

    if p["provider"] == "webgpu":
        from akas.ops.attention.portable import attention

        attend = attention(p, read)
    kernel = primitive(
        [
            ("q", (r, h * d), "float32"),
            ("kc", shape, dtype),
            ("vc", shape, dtype),
            ("ks", scale_shape, "float32"),
            ("vs", scale_shape, "float32"),
            ("out", r * h * d, "float32"),
            ("positions", r, "int32"),
            ("enabled", r, "int32"),
        ],
        attend,
    )
    # The composed plan owns distinct cache/control/output allocations. This
    # lets Tensor prove position loads uniform across each workgroup.
    return kernel.with_attr("tirx.noalias", True) if p.get("optimized") else kernel
