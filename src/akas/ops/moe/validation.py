"""Independent numerical and state qualification for moe."""


def expert_swiglu_validation(kind, p, signature):
    """Independent gate/up products, preserving the declared operand rounding."""
    if p.get("weight_format") == "fp8":
        return fp8_expert_swiglu_validation(kind, p, signature)
    import numpy as np
    from akas.kernels.contracts import ValidationCase, sample_arguments
    from akas.quantization.weights.reference import sample_parts

    values = sample_arguments(signature)
    experts, top, columns, depth = (p[n] for n in ("experts", "top", "o", "k"))
    values["x"] *= 8
    values["ids"][0] = np.linspace(0, experts - 1, top, dtype="int32")
    values["counts"][:] = 0
    values["counts"][values["ids"][0]] = 1
    if top > 1:
        values["counts"][values["ids"][0, -1]] = 0
    gate = sample_parts(p["weight_parts"], depth, values["w"], seed=31)
    up = sample_parts(p["up_weight_parts"], depth, values["w2"], seed=32)
    dtype = "float16" if p.get("round_inputs", False) else "float32"
    source = values["x"][0].astype(dtype).astype("float64")
    result = np.zeros_like(values["out"])
    for rank, expert in enumerate(values["ids"][0]):
        if not values["counts"][expert]:
            continue
        for start in range(0, columns, 128):
            stop = min(start + 128, columns)
            begin, end = expert * columns + start, expert * columns + stop
            g = (gate(begin, end).astype(dtype).astype("float64") @ source).astype(
                "float32"
            )
            u = (up(begin, end).astype(dtype).astype("float64") @ source).astype(
                "float32"
            )
            result[0, rank, start:stop] = g / (1 + np.exp(-g)) * u
    return ValidationCase(values, {**values, "out": result}, rtol=0.002, atol=0.0002)


def tile_tables(counts, capacity, layouts=5):
    import numpy as np

    experts = np.full((layouts, capacity), -1, dtype="int32")
    rows = np.zeros_like(experts)
    for layout in range(layouts):
        block = 4 << layout
        sizes = (counts + block - 1) // block
        selected = np.repeat(np.arange(len(counts)), sizes)
        local = np.concatenate([np.arange(size) for size in sizes])
        experts[layout, : len(selected)] = selected
        rows[layout, : len(local)] = local
    return experts, rows


def tile_validation(kind, p, signature):
    import numpy as np
    from akas.kernels.contracts import ValidationCase, sample_arguments

    values = sample_arguments(signature)
    ids = (np.arange(p["r"])[:, None] + np.arange(p["top"])[None, :]) % p["experts"]
    if p["r"] > 1:
        ids = ids[:-1]
    values["counts"][:] = np.bincount(ids.reshape(-1), minlength=p["experts"])
    experts, rows = tile_tables(
        values["counts"],
        values["tile_experts"].shape[1],
        values["tile_experts"].shape[0],
    )
    return ValidationCase(
        values, {**values, "tile_experts": experts, "tile_rows": rows}, rtol=0, atol=0
    )


def encoded_validation(kind, p, signature):
    import numpy as np
    from akas.kernels.contracts import ValidationCase, sample_arguments
    from akas.quantization.weights.ggml.validation import sample_weight, decode_rows

    values = sample_arguments(signature)
    rows, top, experts = p["r"], p["top"], p["experts"]
    ids = (np.arange(rows)[:, None] + np.arange(top)[None, :]) % experts
    if "ids" in values:
        values["ids"][:] = ids
    active = np.ones(rows, dtype="int32")
    if rows > 1:
        active[-1] = 0
    counts = np.zeros(experts, dtype="int32")
    routes = np.full((experts, rows), -1, dtype="int32")
    for row in range(rows):
        if active[row]:
            for rank, expert in enumerate(ids[row]):
                routes[expert, counts[expert]] = row * top + rank
                counts[expert] += 1
    if kind == "quantized_routes":
        values["active"][:] = active
        expected = {name: value.copy() for name, value in values.items()}
        expected["counts"], expected["routes"] = counts, routes
        return ValidationCase(values, expected, rtol=0, atol=0)
    values["counts"][:] = counts
    if "routes" in values:
        values["routes"][:] = routes
    if "tile_experts" in values:
        tiles, tile_rows = tile_tables(
            counts, values["tile_experts"].shape[1], values["tile_experts"].shape[0]
        )
        values["tile_experts"][:], values["tile_rows"][:] = tiles, tile_rows
    columns, depth = p["o"], p["k"]
    encoding = p["weight_parts"][0]["options"]["encoding"]
    sample_weight(values["w"], encoding, decode=False)
    dtype = "float16" if p.get("round_inputs", True) else "float32"
    x = values["x"].astype(dtype).astype("float64")
    expected = {
        name: value.copy() if name == "out" else value for name, value in values.items()
    }
    for expert in np.unique(ids[active != 0]):
        owners, ranks = np.where((ids == expert) & (active[:, None] != 0))
        source = x[owners, ranks] if p.get("routed_input") else x[owners]
        for start in range(0, columns, 128):
            stop = min(columns, start + 128)
            decoded = decode_rows(
                values["w"],
                encoding,
                depth,
                expert * columns + start,
                expert * columns + stop,
            )
            decoded = decoded.astype(dtype).astype("float64")
            expected["out"][owners, ranks, start:stop] = source @ decoded.T
    return ValidationCase(values, expected, rtol=0.002, atol=0.0002)


def routed_validation(kind, p, signature):
    import numpy as np
    from akas.kernels.contracts import ValidationCase, sample_arguments

    values = sample_arguments(signature)
    if kind == "biased_router":
        values["logits"][0] = 0  # Exercise bias selection and deterministic ties.
        values["bias"][:] = np.arange(p["experts"])[::-1] % 3 * 0.1
        scores = 1 / (1 + np.exp(-values["logits"].astype("float64")))
        order = np.argsort(-(scores + values["bias"]), axis=1, kind="stable")[
            :, : p["top"]
        ]
        chosen = np.take_along_axis(scores, order, axis=1)
        if p.get("normalize", True):
            chosen /= chosen.sum(axis=1, keepdims=True) + 1e-6
        expected = {name: value.copy() for name, value in values.items()}
        expected["ids"][:] = order
        expected["weights"][:] = chosen * p.get("scale", 1.0)
    else:
        values["active"][:] = 1
        if p["r"] > 1:
            values["active"][-1] = 0
        expected = {name: value.copy() for name, value in values.items()}
        expected["out"][:] = (values["experts"] * values["weights"][:, :, None]).sum(
            axis=1
        )
        expected["out"][values["active"] == 0] = 0
    return ValidationCase(values, expected, rtol=1e-5, atol=1e-6)


def tiles(counts, m, size):
    import numpy as np

    experts = np.full(size, -1, dtype="int32")
    offsets = np.zeros(size, dtype="int32")
    index = 0
    for expert, count in enumerate(counts):
        for offset in range((int(count) + m - 1) // m):
            experts[index], offsets[index] = (expert, offset)
            index += 1
    return (experts, offsets)


def pipeline_validation(kind, p, signature):
    if p.get("packed_input"):
        return packed_validation(kind, p, signature)
    import numpy as np
    from akas.kernels.contracts import ValidationCase, sample_arguments
    from akas.ops.moe.validation import validation as reference
    from akas.quantization._shared.e4m3 import decode_storage

    if kind == "fp8_expert_swiglu":
        return fp8_expert_swiglu_validation(kind, p, signature)

    if kind == "expert_tiles":
        values = sample_arguments(signature)
        values["counts"][:] = 0
        values["counts"][: p["top"]] = p["r"]
        expected = {name: value.copy() for name, value in values.items()}
        expected["tile_experts"], expected["tile_offsets"] = tiles(
            values["counts"], p["block_m"], values["tile_experts"].size
        )
        return ValidationCase(values, expected, rtol=0, atol=0)
    if kind == "expert_merge":
        values = sample_arguments(signature)
        expected = {name: value.copy() for name, value in values.items()}
        result = np.zeros(values["out"].shape, dtype="float32")
        for part in range(p["partitions"]):
            result += values["partial"][:, :, part]
        dtype = next(a["dtype"] for a in signature if a["name"] == "out")
        if dtype == "bfloat16":
            from tensor.runtime.dtypes import decode_bfloat16, encode_bfloat16

            result = decode_bfloat16(encode_bfloat16(result))
        elif dtype == "float16":
            result = result.astype("float16").astype("float32")
        expected["out"] = result
        return ValidationCase(values, expected, rtol=0, atol=0)
    adapted = [
        dict(spec, shape=[p["r"], p["top"], p["o"]]) if spec["name"] == "out" else spec
        for spec in signature
    ]
    if p.get("resident_weights"):
        adapted = [
            dict(spec, dtype="uint8") if spec["name"] == "w" else spec
            for spec in adapted
        ]
    baseline = reference("fp8_experts", p, adapted)
    values, expected = (baseline.arguments, baseline.expected)
    if p.get("resident_weights"):
        values["w"] = decode_storage(values["w"]).astype("float16")
        expected["w"] = values["w"].copy()
    if "tile_experts" in values:
        tables = (
            tile_tables(
                values["counts"],
                values["tile_experts"].shape[1],
                values["tile_experts"].shape[0],
            )
            if p.get("tile_map")
            else tiles(values["counts"], p["block_m"], values["tile_experts"].size)
        )
        for name, array in zip(
            ("tile_experts", "tile_offsets"),
            tables,
        ):
            values[name] = array
            expected[name] = array.copy()
    if p["partitions"] == 1:
        dtype = next(item["dtype"] for item in signature if item["name"] == "out")
        if dtype == "float16":
            expected["out"] = expected["out"].astype("float16").astype("float32")
        elif dtype == "bfloat16":
            from tensor.runtime.dtypes import decode_bfloat16, encode_bfloat16

            expected["out"] = decode_bfloat16(encode_bfloat16(expected["out"]))
        return baseline
    _partial_products(values, expected, p)
    return baseline


def _partial_products(values, expected, p):
    import numpy as np
    from akas.quantization._shared.e4m3 import decode_storage

    shape = (p["r"], p["top"], p["partitions"], p["o"])
    values["out"] = np.zeros(shape, dtype="float32")
    expected["out"] = values["out"].copy()
    activation = decode_storage(values["x"]).astype("float64") * np.repeat(
        values["activation_scales"], 128, axis=-1
    )
    span = p["k"] // p["partitions"]
    groups = (
        enumerate(range(p["experts"]))
        if "counts" in values
        else enumerate(values["experts"])
    )
    for group, expert in groups:
        if expert < 0:
            continue
        if "counts" in values:
            routes = [
                divmod(int(route), p["top"])
                for route in values["routes"][expert, : values["counts"][expert]]
            ]
        else:
            routes = [
                (row, int(rank))
                for row, rank in enumerate(values["routes"][group])
                if rank >= 0
            ]
        weight = (
            values["w"][expert].astype("float64")
            if p.get("resident_weights")
            else decode_storage(values["w"][expert]).astype("float64")
        )
        weight *= np.repeat(
            np.repeat(values["scales"][expert], 128, axis=0), 128, axis=1
        )[: p["o"], : p["k"]]
        for row, rank in routes:
            x = activation[row, rank] if p.get("routed_input") else activation[row]
            for part in range(p["partitions"]):
                segment = slice(part * span, (part + 1) * span)
                expected["out"][row, rank, part] = weight[:, segment] @ x[segment]


def paired_partial_validation(kind, p, signature):
    """Evaluate both partitioned products independently from decoded operands."""
    import numpy as np
    from akas.kernels.contracts import ValidationCase

    adapted = [
        dict(spec, name="out") if spec["name"] == "gate" else spec
        for spec in signature
        if spec["name"] not in ("w2", "scales2", "up")
    ]
    baseline = pipeline_validation("expert_partial", p, adapted)
    values, expected = baseline.arguments, baseline.expected
    if p["partitions"] == 1:
        _partial_products(values, expected, p)
    # Distinct gate/up data prevents a swapped or repeated product from passing.
    values["w2"] = np.flip(values["w"], axis=-2).copy()
    values["scales2"] = np.flip(values["scales"], axis=-1).copy()
    other = {**values, "w": values["w2"], "scales": values["scales2"]}
    other_expected = {}
    _partial_products(other, other_expected, p)
    values["gate"] = values.pop("out")
    values["up"] = other["out"]
    expected["gate"] = expected.pop("out")
    expected["up"] = other_expected["out"]
    expected["w2"], expected["scales2"] = values["w2"], values["scales2"]
    return ValidationCase(values, expected, rtol=baseline.rtol, atol=baseline.atol)


def fp8_expert_swiglu_validation(kind, p, signature):
    """Independent paired products, nonlinear activation and block quantization.

    Integer operands with binary scales keep the products exactly representable.
    Signed operands exercise both branches of the nonlinear activation without an
    ideal F64 GEMM obscuring the separately specified MMA accumulation.
    """
    if p.get("packed_input"):
        return packed_validation(kind, p, signature)
    import numpy as np
    from akas.kernels.contracts import ValidationCase, sample_arguments
    from akas.quantization._shared.e4m3 import decode_storage, encode_storage

    values = sample_arguments(signature)
    quantized_output = kind == "fp8_expert_swiglu"

    def rounded(array, dtype):
        if dtype == "bfloat16":
            from tensor.runtime.dtypes import decode_bfloat16, encode_bfloat16

            return decode_bfloat16(encode_bfloat16(array))
        return array.astype(dtype).astype("float32")

    rows, top, experts, depth, columns = (
        p[n] for n in ("r", "top", "experts", "k", "o")
    )
    rng = np.random.default_rng(831)
    for name in ("x", "w", "w2"):
        values[name][:] = rng.choice(
            np.array([56, 64], dtype="uint8"), values[name].shape
        )
        values[name] |= rng.integers(0, 2, values[name].shape, dtype="uint8") * 128
    for name in ("activation_scales", "scales", "scales2"):
        values[name][:] = 0.125
    values["counts"][:] = 0
    values["routes"][:] = -1
    for row in range(rows - (rows > 1)):
        for rank in range(top):
            expert = (row + rank) % experts
            index = values["counts"][expert]
            values["routes"][expert, index] = row * top + rank
            values["counts"][expert] += 1
    values["tile_experts"], values["tile_offsets"] = (
        tile_tables(
            values["counts"],
            values["tile_experts"].shape[1],
            values["tile_experts"].shape[0],
        )
        if p.get("tile_map")
        else tiles(values["counts"], p["block_m"], values["tile_experts"].size)
    )
    values["out"][:] = 0
    if quantized_output:
        values["output_scales"][:] = 0
    expected = {
        name: value.copy() if name in ("out", "output_scales") else value
        for name, value in values.items()
    }
    normalized = np.zeros(values["out"].shape, dtype="float32")
    source = decode_storage(values["x"]).astype("float64") * 0.125
    for expert, count in enumerate(values["counts"]):
        if not count:
            continue
        owners, ranks = np.divmod(values["routes"][expert, :count], top)
        for start in range(0, columns, 128):
            products = []
            for name in ("w", "w2"):
                weights = (
                    decode_storage(values[name][expert, start : start + 128]).astype(
                        "float64"
                    )
                    * 0.125
                )
                products.append((source[owners] @ weights.T).astype("float32"))
            gate, up = (
                rounded(value, p.get("product_dtype", "float32")) for value in products
            )
            activation = rounded(
                gate / (1 + np.exp(-gate)) * up, p.get("activation_dtype", "float32")
            )
            if quantized_output:
                scales = np.maximum(
                    np.abs(activation).max(axis=1), np.float32(1e-12)
                ) / np.float32(448)
                normalized[owners, ranks, start : start + 128] = (
                    activation / scales[:, None]
                )
                expected["out"][owners, ranks, start : start + 128] = encode_storage(
                    activation / scales[:, None]
                )
                expected["output_scales"][owners, ranks, start // 128] = scales
            else:
                expected["out"][owners, ranks, start : start + 128] = rounded(
                    activation, p.get("output_dtype", "float32")
                )

    def compare_encoding(actual, reference):
        # CPU exp and CUDA exp need not round to the same F32 value. Preserve
        # the F32 tolerance at the activation boundary: a different FP8 code is
        # allowed only when that tiny interval crosses its rounding midpoint.
        # The separate GPU equivalence tests require exact values and scales.
        tolerance = np.abs(normalized) * np.float32(2e-6) + np.float32(1e-8)
        lower = encode_storage(normalized - tolerance)
        upper = encode_storage(normalized + tolerance)
        allowed = (actual == reference) | (actual == lower) | (actual == upper)
        np.testing.assert_array_equal(allowed, np.ones_like(allowed))

    return ValidationCase(
        values,
        expected,
        rtol=2e-6,
        atol=1e-8,
        comparators={"out": compare_encoding} if quantized_output else {},
    )


def validation(kind, p, signature):
    """Independent routing, grouped projection and weighted-combine reference."""
    import numpy as np
    from tensor.runtime.dtypes import decode_bfloat16, encode_bfloat16
    from akas.kernels.contracts import ValidationCase, sample_arguments
    from akas.quantization._shared.e4m3 import decode_storage, encode_storage

    def rounded(x):
        return decode_bfloat16(encode_bfloat16(x))

    values = sample_arguments(signature)
    comparators = {}
    rows, top = (p.get("r", p.get("slots", 1) * p.get("chunk", 1)), p.get("top", 8))
    if kind == "router":
        scores = rounded(values["logits"])
        order = np.argsort(-scores, axis=1, kind="stable")[:, :top]
        selected = np.take_along_axis(scores, order, axis=1)
        probability = np.exp(selected - selected[:, :1])
        expected = {name: value.copy() for name, value in values.items()}
        expected["ids"][:] = order
        expected["weights"][:] = probability / probability.sum(axis=1, keepdims=True)
    elif kind in ("moe_groups", "expert_routes"):
        experts = p.get("experts", max(top + 1, 16))
        values["ids"][:] = (
            np.arange(rows)[:, None] + np.arange(top)[None, :]
        ) % experts
        if "active" in values:
            values["active"][:] = 1
            if rows > 1:
                values["active"][-1] = 0
        expected = {name: value.copy() for name, value in values.items()}
        if kind == "moe_groups":
            expected["experts"][:] = -1
            expected["routes"][:] = -1
            first = {}
            for row in range(rows):
                for rank in range(top):
                    expert = int(values["ids"][row, rank])
                    group = first.setdefault(expert, row * top + rank)
                    expected["experts"][group] = expert
                    expected["routes"][group, row] = rank
        else:
            expected["counts"][:] = 0
            for expert in range(expected["counts"].size):
                routes = np.flatnonzero(
                    (
                        (values["ids"] == expert) & (values["active"][:, None] != 0)
                    ).reshape(-1)
                )
                expected["counts"][expert] = len(routes)
                expected["routes"][expert, : len(routes)] = routes

            def compare_routes(actual, reference):
                for expert, count in enumerate(expected["counts"]):
                    np.testing.assert_array_equal(
                        np.sort(actual[expert, :count]),
                        np.sort(reference[expert, :count]),
                    )
                    np.testing.assert_array_equal(
                        actual[expert, count:], reference[expert, count:]
                    )

            comparators["routes"] = compare_routes
    elif kind == "moe_combine":
        values["weights"][:] = np.abs(values["weights"])
        values["weights"][:] /= values["weights"].sum(axis=1, keepdims=True)
        expected = {name: value.copy() for name, value in values.items()}
        total = np.sum(
            rounded(values["experts"]) * values["weights"][:, :, None], axis=1
        )
        expected["out"][:] = total + rounded(values["shared"]) / (
            1 + np.exp(-rounded(values["gate"]))
        )
    elif kind == "fp8_experts":
        k, o, experts = (p["k"], p["o"], p["experts"])
        rng = np.random.default_rng(37)
        values["w"][:] = rng.integers(32, 56, values["w"].shape, dtype="uint8")
        values["scales"][:] = 0.125
        if "activation_scales" in values:
            values["x"][:] = rng.integers(32, 56, values["x"].shape, dtype="uint8")
            values["activation_scales"][:] = 0.125
            activation = decode_storage(values["x"]) * np.repeat(
                values["activation_scales"], 128, axis=-1
            )
        else:
            blocks = values["x"].reshape(*values["x"].shape[:-1], k // 128, 128)
            scales = np.maximum(
                np.max(np.abs(blocks), axis=-1, keepdims=True), np.float32(1e-12)
            ) / np.float32(448)
            activation = (
                decode_storage(encode_storage(blocks / scales)) * scales
            ).reshape(values["x"].shape)
        values["routes"][:] = -1
        if "counts" in values:
            values["counts"][:] = 0
            for row in range(rows):
                for rank in range(top):
                    expert = (row + rank) % experts
                    index = values["counts"][expert]
                    values["routes"][expert, index] = row * top + rank
                    values["counts"][expert] += 1
        else:
            values["experts"][:] = -1
            for row in range(rows):
                for rank in range(top):
                    group = row * top + rank
                    values["experts"][group] = group % experts
                    values["routes"][group, row] = rank
        expected = {name: value.copy() for name, value in values.items()}
        groups = (
            enumerate(range(experts))
            if "counts" in values
            else enumerate(values["experts"])
        )
        for group, expert in groups:
            if expert < 0:
                continue
            if "counts" in values:
                routes = [
                    (int(value) // top, int(value) % top)
                    for value in values["routes"][expert, : values["counts"][expert]]
                ]
                if not routes:
                    continue
            else:
                routes = [
                    (row, rank)
                    for row, rank in enumerate(values["routes"][group])
                    if rank >= 0
                ]
            weight = decode_storage(values["w"][expert]).astype("float64")
            weight *= np.repeat(
                np.repeat(values["scales"][expert], 128, axis=0), 128, axis=1
            )[:o, :k]
            owners, ranks = np.asarray(routes, dtype="int64").T
            source = (
                activation[owners, ranks]
                if p.get("routed_input")
                else activation[owners]
            )
            expected["out"][owners, ranks] = source.astype("float64") @ weight.T
    else:
        raise ValueError(f"missing mixture-of-experts validation: {kind}")
    return ValidationCase(values, expected, comparators=comparators)


VALIDATIONS = {
    name: validation
    for name in ("router", "moe_groups", "expert_routes", "fp8_experts", "moe_combine")
}


def pack_validation(kind, p, signature):
    """Byte-for-byte routed packing, aligned padding and reusable offsets."""
    import numpy as np
    from akas.kernels.contracts import ValidationCase, sample_arguments

    values = sample_arguments(signature)
    rows, top, experts = (p[name] for name in ("r", "top", "experts"))
    alignment = p.get("max_block_m", 64)
    values["counts"][:] = 0
    values["routes"][:] = -1
    # Include empty experts, inactive rows and uneven final tiles.
    for row in range(rows - (rows > 1)):
        for rank in range(top):
            expert = (row * 3 + rank) % experts
            index = values["counts"][expert]
            values["routes"][expert, index] = row * top + rank
            values["counts"][expert] += 1
    values["tile_experts"], values["tile_offsets"] = tile_tables(
        values["counts"],
        values["tile_experts"].shape[1],
        values["tile_experts"].shape[0],
    )
    expected = pack_reference(values, p)
    return ValidationCase(values, expected, rtol=0, atol=0)


def pack_reference(values, p):
    """Independent aligned address mapping shared by pack and product recipes."""
    top = p["top"]
    alignment = p.get("max_block_m", 64)
    expected = {name: value.copy() for name, value in values.items()}
    offset = 0
    for expert, count in enumerate(values["counts"]):
        if not count:
            continue
        expected["expert_offsets"][expert] = offset
        padded = (int(count) + alignment - 1) // alignment * alignment
        expected["packed"][offset : offset + padded] = 0
        expected["packed_scales"][offset : offset + padded] = 1
        for local, route in enumerate(values["routes"][expert, :count]):
            row, rank = divmod(int(route), top)
            index = (row, rank) if p.get("routed_input") else row
            expected["packed"][offset + local] = values["x"][index]
            expected["packed_scales"][offset + local] = values["activation_scales"][
                index
            ]
        offset += padded
    return expected


def packed_validation(kind, p, signature):
    import numpy as np
    from akas.kernels.registry import validation
    from akas.kernels.contracts import ValidationCase
    from akas.ops.moe.definition import tile_capacity

    rows, top, experts, depth, columns = (
        p[name] for name in ("r", "top", "experts", "k", "o")
    )
    alignment = p.get("max_block_m", 64)
    capacity = tile_capacity(rows, top, experts, alignment) * alignment
    shape = (rows, top, depth) if p.get("routed_input") else (rows, depth)
    adapted = []
    for item in signature:
        name = item["name"]
        if name == "expert_offsets":
            continue
        if name in ("x", "activation_scales"):
            item = {
                **item,
                "shape": [*shape[:-1], depth if name == "x" else depth // 128],
            }
        elif p.get("packed_output") and name in ("out", "output_scales"):
            item = {
                **item,
                "shape": [rows, top, columns if name == "out" else columns // 128],
            }
        adapted.append(item)
    baseline = validation(
        kind, {**p, "packed_input": False, "packed_output": False}, adapted
    )
    values, expected = baseline.arguments, baseline.expected
    packed = pack_reference(
        {
            **values,
            "packed": np.zeros((capacity, depth), "uint8"),
            "packed_scales": np.ones((capacity, depth // 128), "float32"),
            "expert_offsets": np.zeros(experts, "int32"),
        },
        p,
    )
    for name, target in [
        ("x", "packed"),
        ("activation_scales", "packed_scales"),
        ("expert_offsets", "expert_offsets"),
    ]:
        values[name] = packed[target]
        expected[name] = packed[target].copy()
    if p.get("packed_output"):
        for name, width in [("out", columns), ("output_scales", columns // 128)]:
            if name not in values:
                continue
            old = expected[name]
            values[name] = np.zeros((capacity, width), values[name].dtype)
            expected[name] = values[name].copy()
            for expert, count in enumerate(values["counts"]):
                begin = int(values["expert_offsets"][expert])
                for local, route in enumerate(values["routes"][expert, :count]):
                    row, rank = divmod(int(route), top)
                    expected[name][begin + local] = old[row, rank]
    return ValidationCase(values, expected, rtol=baseline.rtol, atol=baseline.atol)
