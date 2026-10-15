"""WebGPU specializations and shared bindings for linear."""

from collections.abc import Mapping
from akas.kernels.definition import KernelSpec
from akas.kernels.kernel import Tensor, kernel, output


def linear(spec: KernelSpec, schedule: Mapping[str, object]):
    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    r, k, o = (p["r"], p["k"], p["o"])
    threads = p.get("threads", 64)
    if threads not in (32, 64, 128, 256):
        raise ValueError("reference projection threads must be 32/64/128/256")
    stride, offset = (p.get("output_stride", o), p.get("output_offset", 0))
    matrix = p.get("matrix_output", False)
    shape = (r, stride) if matrix else (r * stride,)

    @kernel
    def compute(
        x: Tensor((r * k,), "float32"), w: Tensor((o * k,), "float32")
    ) -> Tensor(shape, "float32"):
        out = output(shape, "float32")
        with T.Kernel(T.ceildiv(o, threads), r, threads=threads) as (block, row):
            tx = T.get_thread_binding()
            col = block * threads + tx
            if col < o:
                accum = T.alloc_var("float32")
                accum = 0
                for channel in T.serial(k):
                    accum += x[row * k + channel] * w[col * k + channel]
                if matrix:
                    out[row, offset + col] = accum
                else:
                    out[row * stride + offset + col] = accum
        return out

    return compute


def _packed(p, paired=False, add_residual=False):
    import tilelang.language as T
    from tensor.compiler.webgpu.schedules.outer_product import outer_product_matmul
    from tensor.compiler.webgpu.schedules.partitioned import partitioned_matmul
    from tensor.compiler.webgpu.schedules.register_matmul import register_matmul
    from akas.ops.linear.kernel import _epilogue as projection_epilogue
    from akas.ops.linear.kernel import webgpu_abi as projection_abi
    from akas.quantization._shared.rounding import round_half
    from akas.quantization.weights.ggml.kernel import weight_decoder

    r, k, o, q = (p[n] for n in ("r", "k", "o", "type"))
    parts = 2 if paired else 1
    tm, tn, bk = p.get("tile", (16, 32, 64))
    rounded, decoder = (round_half(), weight_decoder(q, k, packed_words=True))
    epilogue = projection_epilogue(paired=paired, add_residual=add_residual)

    @T.macro
    def read_rhs(w, col, channel):
        if q == 1:
            return decoder(w, col, channel)
        else:
            return rounded(decoder(w, col, channel))

    if p.get("schedule") == "outer":
        if q not in (1, 2, 14):
            raise ValueError("outer-product prefill requires F16, Q4_0 or Q6_K weights")
        dtype = p.get("outer_shared_dtype", "float16")
        algorithm = outer_product_matmul(
            r, k, o, rounded, read_rhs, epilogue, parts=parts, dtype=dtype, **p["outer"]
        )
    elif p.get("schedule") == "partitioned":
        if q != 1:
            raise ValueError("searched partitioned profile requires F16 weights")
        config = {
            name: p[name]
            for name in (
                "threads",
                "partitions",
                "unroll",
                "dot_width",
                "owner_axis",
                "k_layout",
            )
        }
        algorithm = partitioned_matmul(
            r,
            k,
            o,
            rounded,
            read_rhs,
            epilogue,
            parts=parts,
            tile_m=tm,
            tile_n=tn,
            explicit_unroll=p.get("explicit_unroll", False),
            **config,
        )
    else:
        algorithm = register_matmul(
            r,
            k,
            o,
            rounded,
            read_rhs,
            epilogue,
            parts=parts,
            tile_m=tm,
            tile_n=tn,
            tile_k=bk,
            threads=p.get("threads", 128),
            pad=p.get("pad", 0),
            lhs_pad=p.get("lhs_pad", 0),
            lhs_transpose=p.get("lhs_transpose", False),
            dot_width=p.get("dot_width", 4 if q == 1 or k >= 2048 else 1),
            unroll=p.get("unroll", False),
            explicit_unroll=p.get("explicit_unroll", False),
        )
    return projection_abi(p, algorithm, paired=paired, add_residual=add_residual)


def _integer(p, paired=False, add_residual=False):
    from tensor.compiler.webgpu.schedules.packed_integer import packed_integer_matmul
    from akas.ops.linear.kernel import _epilogue as projection_epilogue

    r, k, o = (p.get("r", 1), p["k"], p["o"])
    parts = 2 if paired else 1
    from akas.quantization.weights.ggml.webgpu import integer_rhs

    prepacked = p.get("q4_prepacked", False)
    storage, read_word, read_scale = integer_rhs((o, k), prepacked=prepacked)
    count = storage.values.shape[0]

    config = dict(p.get("integer", {}))
    if prepacked:
        config["signed_rhs"] = True
    algorithm = packed_integer_matmul(
        r,
        k,
        o,
        read_word,
        read_scale,
        projection_epilogue(paired=paired, add_residual=add_residual),
        parts=parts,
        **config,
    )
    if parts == 2:

        @kernel
        def compute(
            packed: Tensor(r * k // 2, "uint32"),
            scales: Tensor(r * k // 16, "float32"),
            sums: Tensor(r * k // 16, "int32"),
            w: Tensor(count, "uint32"),
            w2: Tensor(count, "uint32"),
        ) -> Tensor(r * o, "float32"):
            out = output(r * o, "float32")
            algorithm(packed, scales, sums, w, out, w2)
            return out

        return compute

    @kernel
    def compute(
        packed: Tensor(r * k // 2, "uint32"),
        scales: Tensor(r * k // 16, "float32"),
        sums: Tensor(r * k // 16, "int32"),
        w: Tensor(count, "uint32"),
    ) -> Tensor(r * o, "float32"):
        out = output(r * o, "float32")
        algorithm(packed, scales, sums, w, out)
        return out

    return compute


def gemv_packed(p, paired=False, add_residual=False):
    import tilelang.language as T
    from tensor.compiler.webgpu.schedules.gemv import streamed_gemv
    from tensor.compiler.webgpu.schedules.registers import registers, shared_buffers
    from akas.ops.linear.kernel import _epilogue as projection_epilogue
    from akas.ops.linear.kernel import webgpu_abi as projection_abi

    k, o, q = (p["k"], p["o"], p["type"])
    parts = 2 if paired else 1
    epilogue = projection_epilogue(paired=paired, add_residual=add_residual)
    if p.get("decode_schedule") == "streamed":
        if q != 1:
            raise ValueError("streamed GEMV requires native F16 weights")
        config = {
            name: p[name]
            for name in (
                "lanes",
                "threads",
                "micro_rows",
                "dot_width",
                "unroll",
                "accumulators",
                "k_layout",
                "shared_input",
            )
        }
        return projection_abi(
            p,
            streamed_gemv(k, o, epilogue, parts=parts, **config),
            paired=paired,
            add_residual=add_residual,
        )
    wide = q == 2 and p.get("sg") and (k % 256 == 0) and (min(k, o) >= 2048)
    lanes, threads, chains = (16 if q == 2 else 32, 128, 1)
    fast_q4 = q == 2 and k % 128 == 0
    if fast_q4:
        lanes, threads, chains = (
            p.get("gemv_lanes", 32 if wide else 16),
            p.get("gemv_threads", 128),
            p.get("gemv_accumulators", 1),
        )
        if (
            lanes not in (8, 16, 32, 64)
            or threads not in (64, 128, 256, 512)
            or threads % lanes
            or (chains not in (1, 4))
            or k % (lanes * 8)
        ):
            raise ValueError("unsupported packed Q4 decode schedule")
    if q == 14:
        threads = p.get("gemv_threads", 128)
        if threads not in (64, 128, 256, 512):
            raise ValueError("invalid Q6 decode workgroup")
    row_count = threads // lanes
    tiles = (
        k // (lanes * 8)
        if fast_q4
        else k // 256
        if q == 14
        else k // 128
        if q in (0, 1) and k % 128 == 0
        else k // 32
    )
    unroll, independent = (p.get("gemv_unroll", 1), p.get("gemv_chains", 1))
    dot_q4, dot_q6 = (
        fast_q4 and p.get("gemv_dot", wide),
        q == 14 and p.get("gemv_q6_dot"),
    )
    from akas.quantization.weights.ggml.webgpu import gemv_terms

    compute = gemv_terms(
        p,
        fast_q4=fast_q4,
        lanes=lanes,
        row_count=row_count,
        dot_q4=dot_q4,
        dot_q6=dot_q6,
    )
    if dot_q4 and chains != 1:
        raise ValueError("packed dot schedule uses one accumulator")
    if unroll != 1 or independent != 1:
        if (
            q not in (2, 14)
            or type(unroll) is not int
            or unroll not in (1, 2, 4, 8, 16)
            or (type(independent) is not int)
            or (independent not in (1, 2, 4, 8))
            or (independent > unroll)
            or (chains != 1)
            or tiles % unroll
        ):
            raise ValueError("invalid packed decode unroll or accumulator chains")
        chains = independent

    def compute_slots(acc, x, weights, chunk, lane, row, bx):
        for slot in range(unroll):
            for part in range(parts):
                compute(
                    {j: acc[part, j] for j in range(chains)},
                    x,
                    weights[part],
                    chunk * unroll + slot,
                    lane,
                    row,
                    bx,
                    slot % independent,
                )

    def combine(acc, totals):
        for part in range(parts):
            if chains == 4:
                combine4(totals[part,][0], *(acc[part, j][0] for j in range(4)))
            else:
                for j in range(chains):
                    add(totals[part,][0], acc[part, j][0])

    def shuffles(totals):
        for stride in (lanes >> i for i in range(1, lanes.bit_length())):
            for part in range(parts):
                shuffle(totals[part,][0], stride)

    def spills(totals, scratch, tx):
        for part in range(parts):
            spill(scratch[part], totals[part,][0], tx)

    def steps(scratch, tx, stride):
        for part in range(parts):
            shared_step(scratch[part], tx, stride)

    def reduction(scratch, tx, lane):
        for stride in (lanes >> i for i in range(1, lanes.bit_length())):
            reduction_step(scratch, tx, lane, stride)

    @T.macro
    def combine4(out: T.Ref, a, b, c, d):
        out = a + b + (c + d)

    @T.macro
    def add(out: T.Ref, value):
        out = out + value

    @T.macro
    def shuffle(acc: T.Ref, stride):
        acc = acc + T.call_extern(
            "float32", "subgroupShuffleXor", acc, T.uint32(stride)
        )

    @T.macro
    def spill(scratch, value, tx):
        scratch[tx] = value

    @T.macro
    def shared_step(scratch, tx, stride):
        scratch[tx] = scratch[tx] + scratch[tx + stride]

    @T.macro
    def reduction_step(scratch, tx, lane, stride):
        if lane < stride:
            steps(scratch, tx, stride)
        T.sync_threads()

    @T.macro
    def gemv(x, w, out, w2=None, residual=None):
        with T.Kernel(T.ceildiv(o, row_count), threads=threads) as bx:
            tx = T.get_thread_binding()
            row = tx // lanes
            lane = tx % lanes
            accum = registers((parts, chains))
            totals = registers((parts,))
            scratch = shared_buffers((threads,), "float32", parts)
            if bx * row_count + row < o:
                for chunk in T.serial(tiles // unroll):
                    compute_slots(accum, x, (w, w2), chunk, lane, row, bx)
            combine(accum, totals)
            if p.get("sg"):
                if T.call_extern("uint32", "tensor_subgroup_size") >= lanes:
                    shuffles(totals)
                    spills(totals, scratch, tx)
                else:
                    spills(totals, scratch, tx)
                    T.sync_threads()
                    reduction(scratch, tx, lane)
            else:
                spills(totals, scratch, tx)
                T.sync_threads()
                reduction(scratch, tx, lane)
            if (lane == 0) & (bx * row_count + row < o):
                epilogue(
                    out,
                    residual,
                    bx * row_count + row,
                    scratch[0][tx],
                    scratch[1][tx] if parts == 2 else 0,
                )

    return projection_abi(p, gemv, paired=paired, add_residual=add_residual)


def q8(spec: KernelSpec, schedule: Mapping[str, object]):
    from akas.quantization.weights.ggml.webgpu import q8_dot

    p = {**spec.parameters, **schedule}

    k, o = p["k"], p["o"]
    product = q8_dot(k, o)

    @kernel
    def compute(
        x: Tensor((k // 4,), "uint32"),
        scales: Tensor((k // 32,), "float32"),
        sums: Tensor((k // 32,), "int32"),
        w: Tensor((k * o // 32 * 18 // 4,), "uint32"),
    ) -> Tensor((o,), "float32"):
        out = output((o,), "float32")
        product(x, scales, sums, w, out)
        return out

    return compute


def ffn(spec: KernelSpec, schedule: Mapping[str, object]):
    from akas.ops.linear.webgpu import _integer as integer_prefill_kernel
    from akas.ops.linear.webgpu import _packed as prefill_kernel
    from akas.ops.linear.webgpu import gemv_packed as decode_kernel

    p = {**spec.parameters, **schedule}
    if p.get("q16"):
        return integer_prefill_kernel(p, paired=True, add_residual=False)
    r = p.get("r", 1)
    return (
        decode_kernel(p, paired=True, add_residual=False)
        if r == 1
        else prefill_kernel(p, paired=True, add_residual=False)
    )


def ffn_q16(spec: KernelSpec, schedule: Mapping[str, object]):
    from akas.ops.linear.webgpu import _integer as integer_prefill_kernel

    p = {**spec.parameters, **schedule}
    return integer_prefill_kernel(p, paired=True, add_residual=False)


def kernel_linear(spec: KernelSpec, schedule: Mapping[str, object]):
    from akas.ops.linear.webgpu import _integer as integer_prefill_kernel
    from akas.ops.linear.webgpu import _packed as prefill_kernel
    from akas.ops.linear.webgpu import gemv_packed as decode_kernel

    p = {**spec.parameters, **schedule}
    if p.get("q16"):
        return integer_prefill_kernel(p, paired=False, add_residual=False)
    r = p.get("r", 1)
    return (
        decode_kernel(p, paired=False, add_residual=False)
        if r == 1
        else prefill_kernel(p, paired=False, add_residual=False)
    )


def linear_add(spec: KernelSpec, schedule: Mapping[str, object]):
    from akas.ops.linear.webgpu import _integer as integer_prefill_kernel
    from akas.ops.linear.webgpu import _packed as prefill_kernel
    from akas.ops.linear.webgpu import gemv_packed as decode_kernel

    p = {**spec.parameters, **schedule}
    r = p.get("r", 1)
    if r != 1:
        raise ValueError("residual projection fusion requires decode rows")
    return (
        decode_kernel(p, paired=False, add_residual=True)
        if r == 1
        else prefill_kernel(p, paired=False, add_residual=True)
    )


def linear_q16(spec: KernelSpec, schedule: Mapping[str, object]):
    from akas.ops.linear.webgpu import _integer as integer_prefill_kernel

    p = {**spec.parameters, **schedule}
    return integer_prefill_kernel(p, paired=False, add_residual=False)


def linear_q8(spec: KernelSpec, schedule: Mapping[str, object]):
    from akas.kernels.definition import KernelSpec
    from akas.ops.linear.webgpu import q8 as q8_kernel

    p = {**spec.parameters, **schedule}
    r = p.get("r", 1)
    if r != 1:
        raise ValueError(
            "packed Q8 dot requires one row; use quantized_linear for other shapes"
        )
    return q8_kernel(KernelSpec("q8", p), {})


def floating_ffn(spec: KernelSpec, schedule: Mapping[str, object]):
    p = {**spec.parameters, **schedule}
    from akas.compilation.transforms.projection import (
        batch_rows,
        column_tiles,
        output_view,
        projection_algorithm,
    )
    from akas.ops.linear.webgpu import _packed as prefill_kernel
    from akas.ops.linear.webgpu import gemv_packed as decode_kernel

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
        kernel = decode_kernel({**parameters, "r": 1}, paired=True, add_residual=False)
        if p["r"] > 1:
            kernel = batch_rows(kernel, p["r"], p["k"], p["o"])
    else:
        kernel = prefill_kernel(parameters, paired=True, add_residual=False)
        if parameters.get("schedule") != "outer":
            kernel = column_tiles(kernel)
    return output_view(kernel, p)


def floating_linear(spec: KernelSpec, schedule: Mapping[str, object]):
    p = {**spec.parameters, **schedule}
    from akas.compilation.transforms.projection import (
        batch_rows,
        column_tiles,
        output_view,
        projection_algorithm,
    )
    from akas.ops.linear.webgpu import _packed as prefill_kernel
    from akas.ops.linear.webgpu import gemv_packed as decode_kernel

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
        kernel = decode_kernel({**parameters, "r": 1}, paired=False, add_residual=False)
        if p["r"] > 1:
            kernel = batch_rows(kernel, p["r"], p["k"], p["o"])
    else:
        kernel = prefill_kernel(parameters, paired=False, add_residual=False)
        if parameters.get("schedule") != "outer":
            kernel = column_tiles(kernel)
    return output_view(kernel, p)


MINIMUM_MMA_GROUPS = 512
from .kernel import linear as shared_linear, packed_linear
from .definition import IMPLEMENTATIONS as _CHOICES

IMPLEMENTATIONS = tuple((choice for choice in _CHOICES if choice.provider == "webgpu"))


def quantized(spec, schedule):
    from .kernel import quantized_linear

    return quantized_linear(spec, schedule, provider="webgpu")


def projection(spec, schedule):
    from .kernel import projection

    return projection(spec, schedule, provider="webgpu")
