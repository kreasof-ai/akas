"""Operation kernels for vocabulary.cuda.packed_baseline."""

from akas.ops.quantization.ggml import weight_storage
from akas.ops.quantization.ggml import weight_decoder

from akas.checkpoints.gguf import TYPES


def make_kernel(kind, p):
    import tilelang.language as T

    c = p.get("c")
    r = p.get("r", 1)
    if kind == "embedding":
        q, v = (p["type"], p["v"])
        count, dtype = weight_storage(q, c * v)
        read_weight = weight_decoder(q, c)
        _, block, size = TYPES[q]

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
                        out[row * c + col] = read_weight(w, tokens[row], col)

        return kernel
    raise ValueError("unsupported operation: " + kind)
