"""Two signed Q8 planes reconstruct half-rounded activations for Q4 matmul."""

import numpy as np


def encode(values, *, fixed_residual=False):
    values = np.asarray(values, dtype="float32").astype("float16").astype("float32")
    residual = values.reshape(-1, 32).copy()
    planes, scales, sums = [], [], []
    first = None
    for component in range(2):
        factor = (first / np.float32(254) if component and fixed_residual
                  else np.maximum(np.max(np.abs(residual), axis=1), np.float32(1e-20)*127)
                  / np.float32(127))
        quant = np.rint(residual / factor[:, None]).astype("int8")
        planes.append(quant)
        scales.append(factor)
        sums.append(quant.astype("int32").sum(axis=1, dtype="int32"))
        residual -= quant.astype("float32") * factor[:, None]
        first = factor
    return (np.concatenate(planes).reshape(-1).view("uint32"),
            np.concatenate(scales), np.concatenate(sums))


def decode(packed, scales, *, rows, depth):
    quant = packed.view("int8").reshape(2, rows, depth//32, 32).astype("float64")
    factors = scales.reshape(2, rows, depth//32, 1).astype("float64")
    return np.sum(quant * factors, axis=0).reshape(rows, depth)


def validation(kind, parameters, signature):
    from akas.ops.contracts import ValidationCase, sample_arguments
    from akas.quantization.weights.ggml import sample_weight

    values = sample_arguments(signature)
    rows, depth = parameters["r"], parameters["k"]
    if kind == "quantize_q16":
        packed, scales, sums = encode(values["x"], fixed_residual=parameters.get("fixed_residual", False))
        expected = {name: value.copy() for name, value in values.items()}
        for name, value in (("packed", packed), ("scales", scales), ("sums", sums)):
            expected[name][:] = value.reshape(expected[name].shape)
        # Backend division and residual multiply/subtract may differ by a few
        # float32 ulps. Bound the residual scale error by the first-plane scale;
        # quantized bytes and integer sums still require exact equality.
        def check_scales(actual, wanted):
            baseline = expected["scales"].reshape(2,-1)[0]
            bound = np.tile(baseline * np.finfo("float32").eps * 4, 2)
            assert np.all(np.abs(actual.reshape(-1)-wanted.reshape(-1)) <= bound)

        return ValidationCase(values, expected, comparators={"scales": check_scales})
    rng = np.random.default_rng(43)
    packed, scales, sums = encode(rng.normal(0, .15, (rows, depth)).astype("float32"),
                                  fixed_residual=parameters.get("integer", {}).get("fixed_residual", False))
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


projection_validation = validation

from akas.ops.registry import variants

IMPLEMENTATIONS = tuple(
    item
    for contract in ("reference", "packed")
    for item in variants(("quantize_q16",), "akas.quantization.activations.packing", contract,
                         provider="webgpu", priority=20, factory="quantize_kernel",
                         semantic_keys=("r", "k"), space=(("sg", (False, True)),))
)
VALIDATIONS = {"quantize_q16": validation}

from akas.ops.contracts import OutputABI
OUTPUT_ARGUMENTS = {"*": {"quantize_q16": OutputABI((1, 2, 3), ("packed", "scales", "sums"))}}
