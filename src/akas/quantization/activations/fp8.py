"""Native SM89 FP8 tensor-core projections with block-128 dynamic scales."""


def quantize_kernel(p):
    import tilelang.language as T
    from tensor.compiler.frontend.source import primitive

    r, k, top = p["r"], p["k"], p.get("top")
    rows = r * (top or 1)
    shape = (r, top, k) if top else (r, k)
    scales_shape = (r, top, k // 128) if top else (r, k // 128)

    @T.macro
    def algorithm(x, out, scales):
        with T.Kernel(rows, k // 128, threads=128) as (b, tile):
            values = T.alloc_fragment((128,), "float32")
            absolute = T.alloc_fragment((128,), "float32")
            maximum = T.alloc_fragment((1,), "float32")
            for j in T.Parallel(128):
                if top:
                    values[j] = T.cast(x[b // top, b % top, tile * 128 + j], "float32")
                else:
                    values[j] = T.cast(x[b, tile * 128 + j], "float32")
                absolute[j] = T.abs(values[j])
            T.reduce_max(absolute, maximum, dim=0)
            for j in T.Parallel(128):
                scale = T.max(maximum[0], T.float32(1e-12)) / T.float32(448)
                bits = T.cast(
                    T.call_extern("uint32", "tensor_encode_e4m3", values[j] / scale),
                    "uint8",
                )
                if top:
                    out[b // top, b % top, tile * 128 + j] = bits
                else:
                    out[b, tile * 128 + j] = bits
            for j in T.Parallel(1):
                scale = T.max(maximum[0], T.float32(1e-12)) / T.float32(448)
                if top:
                    scales[b // top, b % top, tile] = scale
                else:
                    scales[b, tile] = scale

    return primitive(
        [
            ("x", shape, "bfloat16"),
            ("out", shape, "uint8"),
            ("scales", scales_shape, "float32"),
        ],
        algorithm,
    )


def make_kernel(kind, parameters):
    if kind != "quantize":
        raise ValueError("unsupported activation codec operation: " + kind)
    return quantize_kernel(parameters)


def validation(kind, parameters, signature):
    import numpy as np
    from akas.ops.contracts import ValidationCase, sample_arguments
    from akas.quantization.weights.block_fp8 import encode_storage

    values = sample_arguments(signature)
    depth = parameters["k"]
    blocks = values["x"].reshape(*values["x"].shape[:-1], depth//128, 128)
    scales = np.maximum(np.max(np.abs(blocks), axis=-1), np.float32(1e-12)) / np.float32(448)
    expected = {name: value.copy() for name, value in values.items()}
    expected["out"][:] = encode_storage(blocks/scales[..., None]).reshape(values["out"].shape)
    expected["scales"][:] = scales
    return ValidationCase(values, expected, rtol=0, atol=1e-9)


from akas.ops.registry import variants

IMPLEMENTATIONS = variants(("quantize",), "akas.quantization.activations.fp8", "block-fp8")
VALIDATIONS = {"quantize": validation}
