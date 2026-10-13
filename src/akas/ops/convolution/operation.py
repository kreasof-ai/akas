"""Compatible kernels and search spaces owned by this operation."""

from akas.ops.registry import variants


IMPLEMENTATIONS = (
    *variants(("conv",), "convolution.cuda.packed_baseline", "packed-baseline"),
    *variants(("conv",), "convolution.cuda.packed_baseline", "packed", priority=0),
    *variants(
        ("conv",),
        "convolution.cuda.packed_baseline",
        "packed",
        provider="webgpu",
        priority=0,
    ),
)

IMPLEMENTATIONS += tuple(
    item for provider in ("cuda", "webgpu")
    for item in variants(('conv',), "convolution.reference", "reference", provider=provider)
)


def validation(kind, p, signature):
    import numpy as np
    from akas.ops.contracts import ValidationCase, sample_arguments

    values = sample_arguments(signature)
    r, c, length, chunk = p["r"], p["c"], p.get("length",3), p.get("chunk",1)
    values["enabled"][:] = chunk
    if r > 1: values["enabled"][-1] = max(0,chunk-1)
    expected = {name: value.copy() for name, value in values.items()}
    x = values["x"].reshape(r,chunk,3,c).astype("float64")
    w = values["w"].reshape(c,length).astype("float64")
    state = expected["state"]
    out = expected["out"].reshape(r,chunk,c)
    for row in range(r):
        for token in range(chunk):
            if token >= values["enabled"][row]:
                out[row,token] = 0
                continue
            current = x[row,token,0] * x[row,token,2]
            acc = current * w[:,-1] + np.sum(state[row].T*w[:,:-1],axis=1)
            out[row,token] = x[row,token,1] * acc
            state[row,:-1] = state[row,1:]
            state[row,-1] = current
    return ValidationCase(values, expected)


VALIDATIONS = {"conv": validation}
