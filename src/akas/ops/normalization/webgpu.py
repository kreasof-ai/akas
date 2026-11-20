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


def affine_layer_norm(spec, schedule):
    """Use subgroup sums for centered variance and keep a row in registers."""
    import tilelang.language as T

    from akas.kernels.kernel import Tensor, kernel, output
    from akas.kernels.webgpu import subgroup_reduce

    p = {**spec.parameters, **schedule}
    rows, width, eps = p["r"], p["c"], p["eps"]
    threads = p.get("threads", 128)
    if width % threads:
        raise ValueError(
            "layer normalization width must divide into complete lane groups"
        )
    count = width // threads
    reduce = subgroup_reduce(threads)

    @kernel
    def compute(
        x: Tensor((rows, width), "float32"),
        scale: Tensor((width,), "float32"),
        bias: Tensor((width,), "float32"),
    ) -> Tensor((rows, width), "float32"):
        out = output((rows, width), "float32")
        with T.Kernel(rows, threads=threads) as row:
            tx = T.get_thread_binding()
            values = T.alloc_local((count,), "float32")
            scratch = T.alloc_shared((threads,), "float32")
            total = T.alloc_var("float32")
            total = 0
            for i in T.unroll(count):
                values[i] = x[row, i * threads + tx]
                total += values[i]
            reduce(scratch, tx, total)
            mean = scratch[0] / width
            total = 0
            for i in T.unroll(count):
                values[i] -= mean
                total += values[i] * values[i]
            T.sync_threads()
            reduce(scratch, tx, total)
            inv_std = T.rsqrt(scratch[0] / width + eps)
            for i in T.unroll(count):
                column = i * threads + tx
                out[row, column] = values[i] * inv_std * scale[column] + bias[column]
        return out

    return compute
