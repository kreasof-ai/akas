"""WebGPU FP8 activations: packed storage with fused software decoding."""

from .definition import FORMAT, IMPLEMENTATIONS as _CHOICES


def dequantize(spec, schedule):
    from .kernel import dequantize

    return dequantize(spec, schedule, provider="webgpu")


def bind(shape):
    from akas.quantization.formats import Codec
    from akas.quantization._shared.access import matrix_tile
    from .kernel import reader

    storage = FORMAT.storage(shape, provider="webgpu")
    read = reader(shape, provider="webgpu")
    return Codec(storage, read, matrix_tile(read, shape))


def quantize(spec, schedule):
    """WebGPU block-128 encoder, matching CUDA's BF16 input semantics."""
    import tilelang.language as T
    from math import prod
    from akas.kernels.kernel import Tensor, kernel, output
    from akas.kernels.reductions import tree_reduce
    from akas.quantization import get
    from akas.quantization._shared.e4m3 import portable_scaling
    from akas.quantization._shared.rounding import round_bfloat

    p = {**spec.parameters, **schedule}
    r, depth, top = p["r"], p["k"], p.get("top")
    shape = (r, top, depth) if top else (r, depth)
    rows = prod(shape[:-1])
    storage = get("activations", "fp8").storage(shape, provider="webgpu")
    values, scale_spec = storage.buffers
    maximum = tree_reduce(128, "max")
    rounded = round_bfloat()
    scale, encode = portable_scaling()

    @kernel
    def compute(
        x: Tensor(shape, "float32"),
    ) -> tuple[
        Tensor(values.shape, values.dtype), Tensor(scale_spec.shape, scale_spec.dtype)
    ]:
        out = output(values.shape, values.dtype)
        scales = output(scale_spec.shape, scale_spec.dtype)
        with T.Kernel(rows, depth // 128, threads=128) as (row, block):
            tx = T.get_thread_binding()
            scratch = T.alloc_shared((128,), "float32")
            source = T.alloc_shared((128,), "float32")
            factor = T.alloc_shared((1,), "float32")
            if top:
                source[tx] = rounded(x[row // top, row % top, block * 128 + tx])
            else:
                source[tx] = rounded(x[row, block * 128 + tx])
            scratch[tx] = T.abs(source[tx])
            T.sync_threads()
            maximum(scratch, tx)
            if tx == 0:
                factor[0] = scale(scratch[0])
                if top:
                    scales[row // top, row % top, block] = factor[0]
                else:
                    scales[row, block] = factor[0]
            T.sync_threads()
            if tx < 32:
                word = T.alloc_var("uint32")
                word = 0
                for component in T.unroll(4):
                    word = word | encode(source[tx * 4 + component], factor[0]) << (
                        component * 8
                    )
                if top:
                    out[row // top, row % top, block * 32 + tx] = word
                else:
                    out[row, block * 32 + tx] = word
        return out, scales

    return compute


AVAILABLE = "webgpu" in dict(FORMAT.providers)
CAPABILITIES = ("dequantize", "read", "read_tile", "quantize")
IMPLEMENTATIONS = tuple(choice for choice in _CHOICES if choice.provider == "webgpu")
