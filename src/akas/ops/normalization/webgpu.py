"""WebGPU specializations and shared bindings for normalization."""

from .kernel import add_rms, kvnorm, qnorm, rms
from .definition import IMPLEMENTATIONS as _CHOICES

IMPLEMENTATIONS = tuple((choice for choice in _CHOICES if choice.provider == "webgpu"))


def post_add_rms(spec, schedule):
    """Keep sandwich normalization and the residual in one subgroup workgroup."""
    import tilelang.language as T
    from akas.kernels.kernel import Tensor, kernel, output
    from akas.kernels.webgpu import subgroup_reduce

    p = {**spec.parameters, **schedule}
    rows, width, eps = p["r"], p["c"], p["eps"]
    threads = p.get("threads", min(256, width))
    if (
        threads not in (32, 64, 128, 256, 512, 1024)
        or width % threads
        or p.get("output_dtype", "float32") != "float32"
    ):
        raise ValueError("WebGPU sandwich normalization requires complete F32 rows")
    reduce = subgroup_reduce(threads)

    @kernel
    def compute(
        x: Tensor((rows * width,), "float32"),
        post_weight: Tensor((width,), "float32"),
        residual: Tensor((rows * width,), "float32"),
        w: Tensor((width,), "float32"),
    ) -> tuple[Tensor((rows * width,), "float32"), Tensor((rows * width,), "float32")]:
        hidden = output((rows * width,), "float32")
        out = output((rows * width,), "float32")
        with T.Kernel(rows, threads=threads) as row:
            tx = T.get_thread_binding()
            values = T.alloc_local((width // threads,), "float32")
            scratch = T.alloc_shared((threads,), "float32")
            total = T.alloc_var("float32")
            scale = T.alloc_var("float32")
            total = 0
            for i in T.unroll(width // threads):
                values[i] = x[row * width + i * threads + tx]
                total += values[i] * values[i]
            reduce(scratch, tx, total)
            scale = T.rsqrt(scratch[0] / width + eps)
            total = 0
            for i in T.unroll(width // threads):
                col = i * threads + tx
                values[i] = (
                    residual[row * width + col] + values[i] * scale * post_weight[col]
                )
                hidden[row * width + col] = values[i]
                total += values[i] * values[i]
            T.sync_threads()
            reduce(scratch, tx, total)
            scale = T.rsqrt(scratch[0] / width + eps)
            for i in T.unroll(width // threads):
                col = i * threads + tx
                out[row * width + col] = values[i] * scale * w[col]
        return hidden, out

    return compute
