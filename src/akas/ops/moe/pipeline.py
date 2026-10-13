"""Physical routed projection stages and compact expert tile maps."""


def make_kernel(kind, p):
    if kind == "expert_tiles":
        from akas.ops.moe.cuda.compact import tile_map_kernel

        return tile_map_kernel(dict(p, rows=p["r"]))
    if kind == "expert_partial":
        if p.get("fixed_rows"):
            from akas.ops.linear.cuda.block_fp8 import make_kernel as projection

            return projection("fp8_experts_mma_prequantized", p)
        if p.get("hopper"):
            from akas.ops.moe.hardware.hopper.experts import expert_kernel
        elif p["partitions"] > 1:
            from akas.ops.moe.cuda.split import expert_kernel
        else:
            from akas.ops.moe.cuda.prefill import expert_kernel
        return expert_kernel(dict(p, rows=p["r"]))
    if kind == "expert_merge":
        from akas.ops.linear.cuda.block_fp8 import merge_kernel

        return merge_kernel(p)
    raise ValueError("unknown routed projection stage")


def tiles(counts, m, size):
    import numpy as np

    experts = np.full(size, -1, dtype="int32")
    offsets = np.zeros(size, dtype="int32")
    index = 0
    for expert, count in enumerate(counts):
        for offset in range((int(count) + m - 1) // m):
            experts[index], offsets[index] = expert, offset
            index += 1
    return experts, offsets


def validation(kind, p, signature):
    import numpy as np
    from akas.ops.contracts import ValidationCase, sample_arguments
    from akas.ops.moe.validation import validation as reference
    from akas.quantization.weights.block_fp8 import decode_storage

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
    values, expected = baseline.arguments, baseline.expected
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
