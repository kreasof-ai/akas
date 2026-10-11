"""TileLang DSL for packed portable and standard-subgroup LFM2 schedules."""

from akas.ops.reductions.webgpu import tree_reduce


def half_bits():
    import tilelang.language as T

    @T.macro
    def unpack(bits):
        return T.call_extern("float32", "tensor_unpack_f16", T.cast(bits, "uint32"))

    return unpack


def packed_loads():
    import tilelang.language as T

    unpack = half_bits()

    @T.macro
    def byte(w, address):
        return (w[address // 4] >> (address % 4 * 8)) & T.uint32(255)

    @T.macro
    def half(w, address):
        return unpack((w[address // 4] >> (address % 4 * 8)) & T.uint32(65535))

    @T.macro
    def word(w, address):
        # The 18-byte GGML Q4 blocks alternate between 2- and 4-byte alignment.
        return T.call_extern(
            "uint32",
            "tensor_byte_align_u32",
            w[address // 4],
            w[(address + 3) // 4],
            T.cast(address % 4, "uint32"),
        )

    return byte, half, word


def round_half():
    """Exact ties-to-even FP16 rounding, including FP16 subnormals.

    Signed exponent comparisons avoid costly unsigned division in WGSL lowering.
    This operation is shared by prefill operand staging and cache stores.
    """
    import tilelang.language as T

    @T.macro
    def rounded(value):
        bits = T.reinterpret("uint32", value)
        exponent = T.cast((bits >> 23) & 255, "int32")
        shift = T.min(T.max(126 - exponent, 1), 24)
        mantissa = (bits & 8388607) | 8388608
        rounding = (
            mantissa + (T.uint32(1) << (shift - 1)) - 1 + ((mantissa >> shift) & 1)
        ) >> shift
        quantum = T.reinterpret("float32", T.uint32(103 << 23))
        small = T.if_then_else(
            exponent < 102, 0.0, T.cast(rounding, "float32") * quantum
        ) * T.if_then_else((bits >> 31) != 0, -1.0, 1.0)
        normal = T.reinterpret(
            "float32", (bits + 4095 + ((bits >> 13) & 1)) & T.uint32(4294959104)
        )
        return T.if_then_else(exponent < 113, small, normal)

    return rounded


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
