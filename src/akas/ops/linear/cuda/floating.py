"""Reuse packed CUDA GEMV/tensor-core GEMM with canonical floating weights."""

MINIMUM_MMA_GROUPS = 32
MAXIMUM_GEMV_ROWS = 31


def mma_block_m(p):
    """Keep two row tiles at the transition from GEMV to tensor cores."""
    ceiling = p.get("maximum_gemv_rows", MAXIMUM_GEMV_ROWS)
    return p.get("block_m", 16 if ceiling < p["r"] <= 32 else 32)


def make_kernel(kind, p):
    from akas.ops.linear.cuda.packed import gemv_kernel, prefill_kernel
    from akas.ops.linear.floating import (
        batch_rows,
        column_tiles,
        output_view,
        projection_algorithm,
    )

    parameters = {**p, "type": 1 if p.get("weight_dtype") == "float16" else 0}
    if (
        projection_algorithm(
            {**p, "maximum_gemv_rows": p.get("maximum_gemv_rows", MAXIMUM_GEMV_ROWS)},
            minimum_groups=p.get("minimum_mma_groups", MINIMUM_MMA_GROUPS),
            default_tile=(32, 64, 32),
        )
        == "gemv"
    ):
        kernel = gemv_kernel(kind, {**parameters, "r": 1})
        if p["r"] > 1:
            kernel = batch_rows(kernel, p["r"], p["k"], p["o"])
    else:
        if p["k"] % p.get("block_k", 32):
            raise ValueError("projection reduction width must be divisible by block_k")
        kernel = column_tiles(
            prefill_kernel(kind, {**parameters, "block_m": mma_block_m(p)})
        )
    return output_view(kernel, p)
