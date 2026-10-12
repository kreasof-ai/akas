"""Reuse portable packed GEMV/GEMM with canonical floating weights."""

MINIMUM_MMA_GROUPS = 512


def make_kernel(kind, p):
    from akas.ops.linear.webgpu.packed import decode_kernel, prefill_kernel
    from akas.ops.linear.floating import (
        batch_rows,
        column_tiles,
        output_view,
        projection_algorithm,
    )

    parameters = {
        **p,
        "type": 1 if p.get("weight_dtype") == "float16" else 0,
        "sg": True,
    }
    if parameters.get("schedule") == "outer":
        parameters.setdefault("outer", {})
    if (
        projection_algorithm(
            p, minimum_groups=p.get("minimum_mma_groups", MINIMUM_MMA_GROUPS)
        )
        == "gemv"
    ):
        kernel = decode_kernel(kind, {**parameters, "r": 1})
        if p["r"] > 1:
            kernel = batch_rows(kernel, p["r"], p["k"], p["o"])
    else:
        kernel = prefill_kernel(kind, parameters)
        if parameters.get("schedule") != "outer":
            kernel = column_tiles(kernel)
    return output_view(kernel, p)
