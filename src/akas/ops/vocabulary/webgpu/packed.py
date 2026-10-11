"""Operation kernels for vocabulary.webgpu.packed."""

from akas.ops.quantization.ggml import weight_decoder
from akas.ops.quantization.ggml import weight_storage


def make_kernel(kind, p):
    import tilelang.language as T

    r = p.get("r", 1)
    if kind == "embedding":
        c, q, v = p["c"], p["type"], p["v"]
        count, dtype = weight_storage(q, v * c, packed_words=True)
        decoder = weight_decoder(q, c, packed_words=True)

        @T.prim_func
        def kernel(
            tokens: T.Tensor((r,), "int32"),
            w: T.Tensor((count,), dtype),
            out: T.Tensor((r * c,), "float32"),
        ):
            with T.Kernel(T.ceildiv(c, 256), r, threads=256) as (bx, row):
                for lane in T.Parallel(256):
                    col = bx * 256 + lane
                    if col < c:
                        out[row * c + col] = decoder(w, tokens[row], col)

        return kernel
    if kind == "argmax":
        # This shared-memory reduction has identical CUDA and WebGPU semantics.
        from akas.ops.vocabulary.cuda.argmax import argmax_kernel

        return argmax_kernel(p, explicit_unroll=True)
    raise ValueError("unsupported operation: " + kind)
