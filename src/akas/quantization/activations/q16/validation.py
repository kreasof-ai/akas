"""Independent numerical and state qualification for q16."""


def encode(values, *, fixed_residual=False):
    import numpy as np

    values = np.asarray(values, dtype="float32").astype("float16").astype("float32")
    residual = values.reshape(-1, 32).copy()
    planes, scales, sums = ([], [], [])
    first = None
    for component in range(2):
        factor = (
            first / np.float32(254)
            if component and fixed_residual
            else np.maximum(np.max(np.abs(residual), axis=1), np.float32(1e-20) * 127)
            / np.float32(127)
        )
        quant = np.rint(residual / factor[:, None]).astype("int8")
        planes.append(quant)
        scales.append(factor)
        sums.append(quant.astype("int32").sum(axis=1, dtype="int32"))
        # GPU uses one explicitly fused F32 residual update. F16 inputs and
        # signed-byte quants make the product/subtraction exact in F64 first.
        residual = (residual.astype("float64")
                    - quant.astype("float64") * factor[:, None].astype("float64")).astype("float32")
        first = factor
    return (
        np.concatenate(planes).reshape(-1).view("uint32"),
        np.concatenate(scales),
        np.concatenate(sums),
    )


def decode(packed, scales, *, rows, depth):
    import numpy as np

    quant = packed.view("int8").reshape(2, rows, depth // 32, 32).astype("float64")
    factors = scales.reshape(2, rows, depth // 32, 1).astype("float64")
    return np.sum(quant * factors, axis=0).reshape(rows, depth)


def validation(kind, parameters, signature):
    import numpy as np
    from akas.kernels.contracts import ValidationCase, sample_arguments
    from akas.quantization.activations.q16.validation import decode, encode

    values = sample_arguments(signature)
    rows, depth = (parameters.get("r", 1), parameters["k"])
    if kind == "dequantize_q16":
        expected = {name: value.copy() for name, value in values.items()}
        expected["out"][:] = decode(
            values["packed"], values["scales"], rows=rows, depth=depth
        )
        return ValidationCase(values, expected, rtol=1e-6, atol=1e-6)
    if kind == "quantize_q16":
        blocks = values["x"].reshape(-1, 32)
        blocks[0] = 0
        if len(blocks) > 1:
            blocks[1, :8] = [127, -127, 0.5, 1.5, 2.5, -0.5, -1.5, -2.5]
        packed, scales, sums = encode(
            values["x"], fixed_residual=parameters.get("fixed_residual", False)
        )
        expected = {name: value.copy() for name, value in values.items()}
        for name, value in (("packed", packed), ("scales", scales), ("sums", sums)):
            expected[name][:] = value.reshape(expected[name].shape)

        def check_scales(actual, wanted):
            baseline = expected["scales"].reshape(2, -1)[0]
            bound = np.tile(baseline * np.finfo("float32").eps * 4, 2)
            assert np.all(np.abs(actual.reshape(-1) - wanted.reshape(-1)) <= bound)

        return ValidationCase(values, expected, comparators={"scales": check_scales})
    raise ValueError(f"unsupported Q16 conversion: {kind}")
