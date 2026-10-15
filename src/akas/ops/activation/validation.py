"""Independent numerical and state qualification for activation."""


def validation(kind, p, signature):
    import numpy as np
    from akas.kernels.contracts import ValidationCase, sample_arguments

    values = sample_arguments(signature)
    p = {
        **p,
        "r": p.get("slots", p.get("r", 1)) * p.get("chunk", 1)
        if p.get("packed")
        else p.get("r", 1),
    }
    expected = {
        name: value.copy() if name == "out" else value for name, value in values.items()
    }
    r, c = (p["r"], p["c"])
    if p.get("expert_rows"):
        r *= p.get("top", 1)
    if len(values) == 2 and "x" in values:
        x = values["x"].reshape(r, 2 * c)
        gate_all, up_all = (x[:, :c], x[:, c:])
    else:
        inputs = [v.reshape(r, c) for name, v in values.items() if name != "out"]
        gate_all, up_all = inputs
    out_bf16 = (
        next((row["dtype"] for row in signature if row["name"] == "out")) == "bfloat16"
    )
    out_2d = expected["out"].reshape(r, c)
    from tensor.runtime.dtypes import decode_bfloat16, encode_bfloat16

    for begin in range(0, r, 1024):
        end = min(r, begin + 1024)
        gate, up = (gate_all[begin:end], up_all[begin:end])
        if p.get("round_inputs"):
            gate = decode_bfloat16(encode_bfloat16(gate))
            up = decode_bfloat16(encode_bfloat16(up))
        gate_f64 = gate.astype("float64")
        out = gate_f64 / (1 + np.exp(-gate_f64)) * up
        if out_bf16:
            out = decode_bfloat16(encode_bfloat16(out))
        out_2d[begin:end] = out
    return ValidationCase(values, expected)
