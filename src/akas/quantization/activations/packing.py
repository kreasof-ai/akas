"""One GPU packing kernel for single and residual Q8 activation planes."""

from akas.ops.reductions.webgpu import tree_reduce
from akas.quantization.weights.storage import round_half


def quantize_kernel(kind, p):
    import tilelang.language as T
    from tensor.compiler.frontend.source import primitive

    r, k = (p.get("r", 1) if kind == "quantize_q16" else 1), p["k"]
    if k % 32:
        raise ValueError("activation blocks require depth divisible by 32")
    blocks, words, components = (
        r * k // 32,
        r * k // 4,
        2 if kind == "quantize_q16" else 1,
    )
    maximum, total_reduce, rounded = (
        tree_reduce(32, "max"),
        tree_reduce(32),
        round_half(),
    )

    def shared_components(
        packed, scales, sums, quants, values, scratch, factor, block, tx
    ):
        for component in range(components):
            component_shared(
                packed,
                scales,
                sums,
                quants,
                values,
                scratch,
                factor,
                block,
                tx,
                component,
            )

    def subgroup_components(packed, scales, sums, value, factor, block, tx):
        for component in range(components):
            component_subgroup(
                packed, scales, sums, value, factor, block, tx, component
            )

    @T.macro
    def component_shared(
        packed,
        scales,
        sums,
        quants,
        values,
        scratch,
        factor: T.Ref,
        block,
        tx,
        component,
    ):
        scratch[tx] = T.abs(values[tx])
        T.sync_threads()
        maximum(scratch, tx)
        if component == 1 and p.get("fixed_residual"):
            factor = scales[block] / 254
        else:
            factor = T.max(scratch[0] / 127, 1.0e-20)
        quants[component, tx] = T.cast(T.round(values[tx] / factor), "int32")
        T.sync_threads()
        if tx < 8:
            packed[block * 8 + tx + component * words] = (
                (T.reinterpret("uint32", quants[component, tx * 4]) & 255)
                | ((T.reinterpret("uint32", quants[component, tx * 4 + 1]) & 255) << 8)
                | ((T.reinterpret("uint32", quants[component, tx * 4 + 2]) & 255) << 16)
                | ((T.reinterpret("uint32", quants[component, tx * 4 + 3]) & 255) << 24)
            )
        if components == 2:
            values[tx] = values[tx] - T.cast(quants[component, tx], "float32") * factor
        scratch[tx] = T.cast(quants[component, tx], "float32")
        T.sync_threads()
        total_reduce(scratch, tx)
        if tx == 0:
            scales[block + component * blocks] = factor
            sums[block + component * blocks] = T.cast(scratch[0], "int32")
        if components == 2:
            T.sync_threads()

    @T.macro
    def component_subgroup(
        packed, scales, sums, value: T.Ref, factor: T.Ref, block, tx, component
    ):
        if component == 1 and p.get("fixed_residual"):
            factor = factor / 254
        else:
            factor = T.max(
                T.call_extern("float32", "subgroupMax", T.abs(value)) / 127, 1.0e-20
            )
        quant = T.cast(T.round(value / factor), "int32")
        bits = (
            (
                T.reinterpret(
                    "uint32",
                    T.call_extern(
                        "int32", "subgroupShuffle", quant, T.uint32((tx // 4) * 4)
                    ),
                )
                & 255
            )
            | (
                (
                    T.reinterpret(
                        "uint32",
                        T.call_extern(
                            "int32",
                            "subgroupShuffle",
                            quant,
                            T.uint32((tx // 4) * 4 + 1),
                        ),
                    )
                    & 255
                )
                << 8
            )
            | (
                (
                    T.reinterpret(
                        "uint32",
                        T.call_extern(
                            "int32",
                            "subgroupShuffle",
                            quant,
                            T.uint32((tx // 4) * 4 + 2),
                        ),
                    )
                    & 255
                )
                << 16
            )
            | (
                (
                    T.reinterpret(
                        "uint32",
                        T.call_extern(
                            "int32",
                            "subgroupShuffle",
                            quant,
                            T.uint32((tx // 4) * 4 + 3),
                        ),
                    )
                    & 255
                )
                << 24
            )
        )
        total = T.call_extern("int32", "subgroupAdd", quant)
        if tx % 4 == 0:
            packed[block * 8 + tx // 4 + component * words] = bits
        if tx == 0:
            scales[block + component * blocks] = factor
            sums[block + component * blocks] = total
        value = value - T.cast(quant, "float32") * factor

    @T.macro
    def quantize(x, packed, scales, sums):
        with T.Kernel(blocks, threads=32) as block:
            tx = T.get_thread_binding()
            scratch = T.alloc_shared((32,), "float32")
            quants = T.alloc_shared((components, 32), "int32")
            values = T.alloc_shared((32,), "float32")
            factor = T.alloc_var("float32")
            if components == 2 and p.get("sg"):
                value = T.alloc_var("float32")
                value = rounded(x[block * 32 + tx])
                if T.call_extern("uint32", "tensor_subgroup_size") >= 32:
                    subgroup_components(packed, scales, sums, value, factor, block, tx)
                else:
                    values[tx] = rounded(x[block * 32 + tx])
                    shared_components(
                        packed, scales, sums, quants, values, scratch, factor, block, tx
                    )
            else:
                values[tx] = (
                    rounded(x[block * 32 + tx])
                    if components == 2
                    else x[block * 32 + tx]
                )
                shared_components(
                    packed, scales, sums, quants, values, scratch, factor, block, tx
                )

    return primitive(
        [
            ("x", r * k, "float32"),
            ("packed" if components == 2 else "out", components * words, "uint32"),
            ("scales", components * blocks, "float32"),
            ("sums", components * blocks, "int32"),
        ],
        quantize,
    )
