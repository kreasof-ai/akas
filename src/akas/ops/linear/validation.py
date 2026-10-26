"""Independent numerical and state qualification for linear."""


def projection_validation(kind, p, signature):
    import numpy as np
    from akas.kernels.contracts import ValidationCase, sample_arguments
    from akas.quantization.weights.ggml.validation import sample_weight
    from akas.quantization._shared.access import restore_order

    values = sample_arguments(signature)
    depth = p["k"]
    weights = []
    for index, part in enumerate(p["weight_parts"]):
        name = "w" if index == 0 else f"weight{index}"
        decoded = sample_weight(
            values[name], part["options"]["encoding"], seed=31 + index
        ).reshape(part["rows"], depth)
        decoded = restore_order(decoded, 0, part.get("row_order"))
        decoded = restore_order(decoded, 1, part.get("column_order"))
        weights.append(decoded)
    x = values["x"]
    expected = {
        name: value.copy() if name == "out" else value for name, value in values.items()
    }
    webgpu_gemv = p["r"] == 1 and any(row["dtype"] == "uint32" for row in signature)
    x_f = x.astype("float16").astype("float64")
    col = 0
    for weight in weights:
        rows = weight.shape[0]
        for start in range(0, rows, 4096):
            stop = min(rows, start + 4096)
            w_slice = weight[start:stop]
            w_part = (
                w_slice.astype("float64")
                if webgpu_gemv
                else w_slice.astype("float16").astype("float64")
            )
            expected["out"][:, col + start : col + stop] = x_f @ w_part.T
        col += rows
    return ValidationCase(values, expected, rtol=0.002, atol=0.002)


def pipeline_validation(kind, p, signature):
    import numpy as np
    from akas.kernels.contracts import ValidationCase, sample_arguments
    from akas.ops.linear.validation import validation as reference
    from akas.quantization._shared.e4m3 import decode_storage

    if kind == "projection_merge":
        values = sample_arguments(signature)
        expected = {name: value.copy() for name, value in values.items()}
        output = np.zeros_like(values["out"])
        for part in range(p["partitions"]):
            output += values["partial"][:, part]
        expected["out"] = output
        return ValidationCase(values, expected, rtol=0, atol=0)
    adapted = [
        dict(spec, shape=[p["r"], p["o"]]) if spec["name"] == "out" else spec
        for spec in signature
    ]
    baseline = reference("fp8_linear", p, adapted)
    values = baseline.arguments
    if p["partitions"] == 1:
        return baseline
    x = decode_storage(values["x"]).astype("float64")
    x *= np.repeat(values["activation_scales"], 128, axis=1)
    weight = decode_storage(values["w"]).astype("float64")
    weight *= np.repeat(np.repeat(values["scales"], 128, axis=0), 128, axis=1)[
        : p["o"], : p["k"]
    ]
    shape = (p["r"], p["partitions"], p["o"])
    values["out"] = np.zeros(shape, dtype="float32")
    expected = {name: value.copy() for name, value in values.items()}
    span = p["k"] // p["partitions"]
    for part in range(p["partitions"]):
        segment = slice(part * span, (part + 1) * span)
        expected["out"][:, part] = x[:, segment] @ weight[:, segment].T
    return ValidationCase(values, expected)


def validation(kind, p, signature):
    if p.get("q16") or kind in ("linear_q16", "ffn_q16"):
        return q16_projection_validation(kind, p, signature)
    import numpy as np
    from akas.kernels.contracts import ValidationCase, sample_arguments

    values = sample_arguments(signature)
    expected = {
        name: value.copy() if name == "out" else value for name, value in values.items()
    }
    r, k, o = (p["r"], p["k"], p["o"])
    if (
        p.get("ggml", False) or p.get("type", 0) not in (0, 1)
    ) and "scales" not in values:
        from akas.quantization.weights.ggml.validation import sample_weight, decode_rows

        x = values["x"].reshape(r, k)
        sample_weight(values["w"], p["type"], decode=False)
        if r > 1:
            x = x.astype("float16").astype("float32")
        if kind == "ffn":
            sample_weight(values["w2"], p["type"], seed=32, decode=False)
        x = x.astype("float64")
        out = np.empty((r, o), dtype="float64")
        half = p.get("half_accum", False)
        if half:
            # Eight F16 FMAs per even/odd lane, then a F32 complete-K sum.
            # Cover both permitted F16 rounding directions, plus subnormal
            # absolute error. Ordinary F32 projections keep their old gate.
            half_gamma = 8 * 2**-10 / (1 - 8 * 2**-10)
            total_gamma = ((k + 15) // 16) * 2**-23
            bounds = np.empty((r, o), dtype="float64")
            absolute_x = np.abs(x)
        for start in range(0, o, 128):
            stop = min(o, start + 128)
            weight = decode_rows(values["w"], p["type"], k, start, stop)
            if r > 1:
                weight = weight.astype("float16")
            section = x @ weight.astype("float64").T
            if half:
                error = ((half_gamma + total_gamma)
                         * (absolute_x @ np.abs(weight.astype("float64")).T)
                         + k * 2**-24)
            if kind == "ffn":
                other = decode_rows(values["w2"], p["type"], k, start, stop)
                if r > 1:
                    other = other.astype("float16")
                up = x @ other.astype("float64").T
                if half:
                    up_error = ((half_gamma + total_gamma)
                                * (absolute_x @ np.abs(other.astype("float64")).T)
                                + k * 2**-24)
                    error = (1.1 * np.abs(up) * error
                             + np.abs(section / (1 + np.exp(-section))) * up_error
                             + 1.1 * error * up_error)
                section = section / (1 + np.exp(-section)) * up
            elif kind == "linear_add":
                section += values["residual"].reshape(r, o)[:, start:stop]
                if half:
                    error += np.abs(section) * 2**-23
            out[:, start:stop] = section
            if half:
                bounds[:, start:stop] = error + np.abs(section) * 2**-21
        expected["out"][:] = out.reshape(values["out"].shape)
        if p.get("weight_layout", "raw") == "planar":
            if p["type"] != 2:
                raise ValueError("planar dense validation requires Q4_0")
            from akas.quantization.weights.ggml.kernel import repack_host
            for name in ("w", "w2") if kind == "ffn" else ("w",):
                raw = values[name].view("uint8").reshape(-1)
                raw[:] = repack_host(raw)
        if half:
            def check_short_half(actual, wanted):
                a, w = actual.astype("float64"), wanted.astype("float64")
                assert np.isfinite(a).all()
                assert np.all(np.abs(a - w) <= bounds.reshape(w.shape))
                # The worst-case per-element cancellation bound alone is too
                # permissive for a broken shader. Retain a separate RMS gate.
                assert np.linalg.norm(a - w) / max(float(np.linalg.norm(w)), 1e-30) < .005
            return ValidationCase(values, expected, comparators={"out": check_short_half})
        return ValidationCase(values, expected)
    if "scales" in values:
        from akas.quantization._shared.e4m3 import decode_storage, encode_storage

        rng = np.random.default_rng(31)
        values["w"][:] = encode_storage(rng.normal(0, 1, values["w"].shape))
        values["scales"][:] = np.float32(0.125)
        weight = decode_storage(values["w"]).astype("float64")
        weight *= np.repeat(np.repeat(values["scales"], 128, axis=0), 128, axis=1)[
            :o, :k
        ]
        if "activation_scales" in values:
            values["x"][:] = encode_storage(rng.normal(0, 1, values["x"].shape))
            values["activation_scales"][:] = np.float32(0.125)
            x = decode_storage(values["x"]) * np.repeat(
                values["activation_scales"], 128, axis=1
            )
        else:
            blocks = values["x"].reshape(r, k // 128, 128)
            scales = np.maximum(
                np.max(np.abs(blocks), axis=2, keepdims=True), np.float32(1e-12)
            ) / np.float32(448)
            x = (decode_storage(encode_storage(blocks / scales)) * scales).reshape(r, k)
        expected = {name: value.copy() for name, value in values.items()}
        expected["out"][:] = x.astype("float64") @ weight.T
        return ValidationCase(values, expected)
    if kind == "split_merge":
        out = np.zeros((r, o), dtype="float32")
        for part in range(p["parts"]):
            out += values["x"][:, part]
        expected["out"][:] = out
        return ValidationCase(values, expected, rtol=0, atol=0)
    if kind == "split_linear":
        block = p["bk"]
        span = ((k + block - 1) // block + p["parts"] - 1) // p["parts"] * block
        compute_dtype = "float32" if max(r * k, o * k) > 4194304 else "float64"
        for part in range(p["parts"]):
            begin, end = (part * span, min(k, (part + 1) * span))
            expected["out"][:, part] = (
                values["x"][:, begin:end].astype(compute_dtype)
                @ values["w"][:, begin:end].astype(compute_dtype).T
            )
        return ValidationCase(values, expected)
    compute_dtype = "float32" if max(r * k, o * k) > 4194304 else "float64"
    x = values["x"].reshape(r, k).astype(compute_dtype)
    w = values["w"].reshape(o, k).astype(compute_dtype)
    out = x @ w.T
    if kind == "ffn":
        up = x @ values["w2"].reshape(o, k).astype(compute_dtype).T
        out = out / (1 + np.exp(-out)) * up
    elif kind == "linear_add":
        out += values["residual"].reshape(r, o)
    stride, offset = (p.get("output_stride", o), p.get("output_offset", 0))
    expected["out"].reshape(r, stride)[:, offset : offset + o] = out
    return ValidationCase(values, expected, rtol=0.02, atol=0.002)


VALIDATIONS = {
    name: validation
    for name in (
        "linear",
        "gemv",
        "ffn",
        "linear_add",
        "bf16_linear",
        "fp8_linear",
        "split_linear",
        "split_merge",
    )
}


def quantized_validation(kind, parameters, signature):
    """Check encoded operands independently, including explicit FP16 staging."""
    import numpy as np
    from akas.kernels.contracts import ValidationCase, sample_arguments
    from akas.quantization._shared.e4m3 import encode_storage, decode_storage
    from akas.quantization.activations.q16.validation import (
        encode as encode_q16,
        decode as decode_q16,
    )
    from akas.quantization.weights.ggml.validation import sample_weight
    from tensor.runtime.dtypes import encode_bfloat16, decode_bfloat16

    p = parameters
    rows, depth, columns = p["r"], p["k"], p["o"]
    values = sample_arguments(signature)
    rng = np.random.default_rng(71)
    activation = p.get("activation_format")
    source = rng.normal(0, 0.15, (rows, depth)).astype("float32")
    if activation in ("q8", "q16"):
        if activation == "q16":
            packed, scales, sums = encode_q16(source)
            x = decode_q16(packed, scales, rows=rows, depth=depth)
        else:
            blocks = source.reshape(-1, 32)
            scales = np.maximum(
                np.abs(blocks).max(-1) / np.float32(127), np.float32(1e-20)
            )
            quant = np.rint(blocks / scales[:, None]).astype("int8")
            packed = quant.reshape(-1).view("uint32")
            sums = quant.astype("int32").sum(-1, dtype="int32")
            x = (quant.astype("float32") * scales[:, None]).reshape(rows, depth)
        for name, data in [
            ("x", packed),
            ("activation_scales", scales),
            ("sums", sums),
        ]:
            if name in values:
                values[name][:] = data.reshape(values[name].shape)
    elif activation == "fp8":
        source = decode_bfloat16(encode_bfloat16(source))
        blocks = source.reshape(rows, depth // 128, 128)
        scales = np.maximum(np.abs(blocks).max(-1), np.float32(1e-12)) / np.float32(448)
        bits = encode_storage(blocks / scales[..., None])
        values["x"].view("uint8")[:] = bits.reshape(values["x"].view("uint8").shape)
        values["activation_scales"][:] = scales
        x = (decode_storage(bits) * scales[..., None]).reshape(rows, depth)
    else:
        values["x"][:] = source
        x = source
    if p.get("weight_format", "ggml") == "fp8":
        bits = rng.integers(0, 127, (columns, depth), dtype="uint8")
        values["w"].view("uint8")[:] = bits.reshape(values["w"].view("uint8").shape)
        scales = decode_bfloat16(
            encode_bfloat16(
                rng.uniform(0.001, 0.005, values["weight_scales"].shape).astype(
                    "float32"
                )
            )
        )
        values["weight_scales"][:] = scales
        if p.get("weight_options", {}).get("scale_layout") == "channel":
            factors = scales
        else:
            factors = np.repeat(np.repeat(scales, 128, axis=0), 128, axis=1)[
                :columns, :depth
            ]
        w = decode_storage(bits) * factors
    else:
        encoding = p.get("weight_options", {}).get("encoding", p.get("type"))
        w = sample_weight(values["w"], encoding).reshape(columns, depth)
    x = x.astype("float16").astype("float64")
    w = w.astype("float16").astype("float64")
    expected = {name: value.copy() for name, value in values.items()}
    expected["out"][:] = x @ w.T
    # Channel-scaled matrices use the same F16 operands/F32 MMA arithmetic as
    # packed projections. Match their absolute tolerance for long reductions
    # near cancellation; block-scaled references keep the existing threshold.
    channel = p.get("weight_options", {}).get("scale_layout") == "channel"
    return ValidationCase(values, expected, rtol=0.002, atol=0.002 if channel else 0.0001)


def q16_projection_validation(kind, parameters, signature):
    import numpy as np
    from akas.kernels.contracts import ValidationCase, sample_arguments
    from akas.quantization.activations.q16.validation import encode, decode
    from akas.quantization.weights.ggml.validation import sample_weight

    values = sample_arguments(signature)
    rows, depth = parameters["r"], parameters["k"]
    rng = np.random.default_rng(43)
    packed, scales, sums = encode(
        rng.normal(0, 0.15, (rows, depth)).astype("float32"),
        fixed_residual=parameters.get("integer", {}).get("fixed_residual", False),
    )
    for name, value in (("packed", packed), ("scales", scales), ("sums", sums)):
        values[name][:] = value.reshape(values[name].shape)
    width = parameters["o"]
    weights = sample_weight(values["w"], 2).reshape(width, depth)
    x = decode(values["packed"], values["scales"], rows=rows, depth=depth)
    out = x @ weights.astype("float64").T
    if kind in ("ffn", "ffn_q16"):
        other = sample_weight(values["w2"], 2, seed=44).reshape(width, depth)
        up = x @ other.astype("float64").T
        out = out / (1 + np.exp(-out)) * up
    expected = {name: value.copy() for name, value in values.items()}
    expected["out"][:] = out.reshape(expected["out"].shape)
    return ValidationCase(values, expected)


def half_input_validation(kind, parameters, signature):
    import numpy as np
    from akas.kernels.contracts import ValidationCase, sample_arguments

    values = sample_arguments(signature)
    edges = np.array([0., -0., 2**-25, -2**-25, 3*2**-25, 2**-24,
                      2**-14 - 2**-25, 1 + 2**-11, 1 + 3*2**-11, 65504.], "float32")
    flat = values["x"].reshape(-1)
    flat[:min(flat.size, edges.size)] = edges[:min(flat.size, edges.size)]
    expected = {name: value.copy() for name, value in values.items()}
    expected["out"][:] = values["x"].astype("float16")

    def exact(actual, reference):
        np.testing.assert_array_equal(actual.view("uint16"), reference.view("uint16"))

    return ValidationCase(values, expected, comparators={"out": exact})


def geglu_projection_validation(kind, p, signature):
    import math
    import numpy as np
    from akas.kernels.contracts import ValidationCase, sample_arguments
    from akas.quantization.weights.ggml.validation import sample_weight, decode_rows

    values = sample_arguments(signature)
    r, k, o = (p[n] for n in ("r", "k", "o"))
    for i, name in enumerate(("w", "w2")):
        sample_weight(values[name], p["type"], seed=31 + i, decode=False)
    x = values["x"].reshape(r, k).astype("float16").astype("float64")
    expected = dict(values)
    result = np.empty((r, o), dtype="float32")
    for start in range(0, o, 512):
        stop = min(o, start + 512)
        gate, up = [
            x
            @ decode_rows(values[n], p["type"], k, start, stop)
            .astype("float16")
            .astype("float64")
            .T
            for n in ("w", "w2")
        ]
        result[:, start:stop] = (
            0.5 * gate * (1 + np.vectorize(math.erf)(gate / 2**0.5)) * up
        )
    expected["out"] = result.reshape(values["out"].shape).astype(values["out"].dtype)
    return ValidationCase(values, expected, rtol=0.002, atol=0.002)
