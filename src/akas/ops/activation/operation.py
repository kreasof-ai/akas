"""SwiGLU semantics, storage policies and one kernel definition."""

from akas.ops.registry import variants

IMPLEMENTATIONS = tuple(
    item
    for provider in ("cuda", "webgpu")
    for contract,defaults in (
        ("packed-baseline", (("flat",True),("input_names",("x","y")))),
        ("packed", (("flat",True),("input_names",("x","y")))),
        ("reference", (("flat",True),("input_names",("x","y")))),
        ("bf16", (("packed",True),("output_dtype","bfloat16"),("round_inputs",True))),
        ("block-fp8", (("output_dtype","bfloat16"),("round_inputs",True))),
    )
    if provider == "cuda" or contract in ("packed-baseline", "packed", "reference")
    for item in variants(("swiglu",), "activation.kernel", contract,
                         provider=provider,
                         defaults=defaults)
)
IMPLEMENTATIONS += variants(
    ("swiglu_experts",), "activation.kernel", "block-fp8",
    defaults=(("expert_rows",True),("output_dtype","bfloat16"),("round_inputs",True)),
)
ROW_INDEPENDENT = ("swiglu",)

def validation(kind, p, signature):
    import numpy as np
    from akas.ops.contracts import ValidationCase, sample_arguments

    values = sample_arguments(signature)
    p = {**p, "r": p.get("slots",p.get("r",1))*p.get("chunk",1) if p.get("packed") else p.get("r",1)}
    expected = {name: value.copy() for name, value in values.items()}
    if len(values) == 2 and "x" in values:
        x = values["x"].reshape(p["r"], 2*p["c"])
        gate, up = np.split(x, 2, axis=1)
    else:
        inputs = [v for name,v in values.items() if name != "out"]
        gate, up = inputs
    if p.get("round_inputs"):
        from tensor.runtime.dtypes import encode_bfloat16, decode_bfloat16
        gate = decode_bfloat16(encode_bfloat16(gate))
        up = decode_bfloat16(encode_bfloat16(up))
    out = gate.astype("float64") / (1 + np.exp(-gate.astype("float64"))) * up
    if next(row["dtype"] for row in signature if row["name"] == "out") == "bfloat16":
        from tensor.runtime.dtypes import encode_bfloat16, decode_bfloat16
        out = decode_bfloat16(encode_bfloat16(out))
    expected["out"][:] = out.reshape(expected["out"].shape)
    return ValidationCase(values, expected)


VALIDATIONS = {name: validation for name in ("swiglu","swiglu_experts")}






# Output locations are part of the operation ABI, independent of compiler passes.
OUTPUT_ARGUMENTS = {"reference": {'swiglu': (-1,)}}


from akas.ops.contracts import OutputABI

for _contract in ("reference", "packed", "packed-baseline", "bf16", "block-fp8"):
    OUTPUT_ARGUMENTS.setdefault(_contract, {}).update({'swiglu': OutputABI((-1,), ('out',)), 'swiglu_experts': OutputABI((-1,), ('out',))})
