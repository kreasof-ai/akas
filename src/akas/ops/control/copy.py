"""Buffer copies are graph operations, independent of model and provider."""


def make_kernel(kind, p):
    if kind != "copy_buffer":
        raise ValueError("unknown copy operation")
    import tilelang.language as T

    n, dtype = p["n"], p["dtype"]

    @T.prim_func
    def kernel(x: T.Tensor((n,), dtype), out: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, 256), threads=256) as block:
            for lane in T.Parallel(256):
                index = block * 256 + lane
                if index < n:
                    out[index] = x[index]

    return kernel


def validation(kind, p, signature):
    from akas.ops.contracts import ValidationCase, sample_arguments

    values = sample_arguments(signature)
    expected = {name: value.copy() for name, value in values.items()}
    expected["out"] = values["x"].copy()
    return ValidationCase(values, expected, rtol=0, atol=0)
