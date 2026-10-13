"""Independent arithmetic and state qualification for moe operations."""


def validation(kind, p, signature):
    """Independent routing, grouped projection and weighted-combine reference."""
    import numpy as np
    from tensor.runtime.dtypes import decode_bfloat16, encode_bfloat16

    from akas.ops.contracts import ValidationCase, sample_arguments
    from akas.quantization.weights.block_fp8 import decode_storage, encode_storage

    def rounded(x):
        return decode_bfloat16(encode_bfloat16(x))

    values = sample_arguments(signature)
    comparators = {}
    rows, top = p.get("r", p.get("slots", 1) * p.get("chunk", 1)), p.get("top", 8)
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
                # Atomic packing may choose any order within an expert bucket.
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
        k, o, experts = p["k"], p["o"], p["experts"]
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
