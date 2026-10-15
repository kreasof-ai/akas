"""Independent numerical and state qualification for moe."""


def encoded_validation(kind, p, signature):
    import numpy as np
    from akas.kernels.contracts import ValidationCase, sample_arguments
    from akas.quantization.weights.ggml.validation import sample_weight

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
    columns, depth = p["o"], p["k"]
    decoded = (
        sample_weight(values["w"], p["weight_parts"][0]["options"]["encoding"])
        .reshape(experts, columns, depth)
        .astype("float16")
        .astype("float64")
    )
    x = values["x"].astype("float16").astype("float64")
    expected = {name: value.copy() for name, value in values.items()}
    for row in range(rows):
        for rank, expert in enumerate(ids[row]):
            if active[row]:
                source = x[row, rank] if p.get("routed_input") else x[row]
                expected["out"][row, rank] = decoded[expert] @ source
    return ValidationCase(values, expected, rtol=0.002, atol=0.0002)


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
    import numpy as np
    from akas.kernels.contracts import ValidationCase, sample_arguments
    from akas.ops.moe.validation import validation as reference
    from akas.quantization._shared.e4m3 import decode_storage

    if kind == "expert_tiles":
        values = sample_arguments(signature)
        values["counts"][:] = 0
        values["counts"][:8] = p["r"]
        expected = {name: value.copy() for name, value in values.items()}
        expected["tile_experts"], expected["tile_offsets"] = tiles(
            values["counts"], p["block_m"], values["tile_experts"].size
        )
        return ValidationCase(values, expected, rtol=0, atol=0)
    if kind == "expert_merge":
        values = sample_arguments(signature)
        expected = {name: value.copy() for name, value in values.items()}
        result = np.zeros_like(values["out"])
        for part in range(p["partitions"]):
            result += values["partial"][:, :, part]
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
        for name, array in zip(
            ("tile_experts", "tile_offsets"),
            tiles(values["counts"], p["block_m"], values["tile_experts"].size),
        ):
            values[name] = array
            expected[name] = array.copy()
    if p["partitions"] == 1:
        return baseline
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
    return baseline


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
            for row, rank in routes:
                x = activation[row, rank] if p.get("routed_input") else activation[row]
                expected["out"][row, rank] = x.astype("float64") @ weight.T
    else:
        raise ValueError(f"missing mixture-of-experts validation: {kind}")
    return ValidationCase(values, expected, comparators=comparators)


VALIDATIONS = {
    name: validation
    for name in ("router", "moe_groups", "expert_routes", "fp8_experts", "moe_combine")
}
