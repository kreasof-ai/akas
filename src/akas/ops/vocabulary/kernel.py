"""Shared TileLang computations for vocabulary."""


def quantized_embedding(spec, schedule, *, provider):
    import tilelang.language as T
    from akas.kernels.kernel import Tensor, kernel, output
    from akas.quantization.weights import bind_parts

    p = {**spec.parameters, **schedule}
    rows, width = p["r"], p["c"]
    dtype = p.get("output_dtype", "bfloat16")
    binding = bind_parts(p["weight_parts"], width, provider=provider)
    storage, read = binding.storage.values, binding.read

    @kernel
    def compute(
        ids: Tensor((rows,), "int32"), w: Tensor(storage.shape, storage.dtype)
    ) -> Tensor((rows, width), dtype):
        out = output((rows, width), dtype)
        with T.Kernel(rows, T.ceildiv(width, 128), threads=128) as (row, tile):
            for lane in T.Parallel(128):
                column = tile * 128 + lane
                if column < width:
                    out[row, column] = T.if_then_else(
                        ids[row] >= 0 and ids[row] < p["vocab"],
                        read((w,), T.max(0, T.min(ids[row], p["vocab"] - 1)), column),
                        0,
                    )
        return out

    return compute


from collections.abc import Mapping
from akas.kernels.definition import KernelSpec
from akas.kernels.kernel import Tensor, kernel, output


def native_argmax(spec: KernelSpec, schedule: Mapping[str, object]):
    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    r, vocab = (p["r"], p["vocab"])

    @kernel
    def compute(
        logits: Tensor((r, vocab), "float32"),
        positions: Tensor((r,), "int32"),
        active: Tensor((r,), "int32"),
    ) -> tuple[Tensor((r,), "int32"), Tensor((r,), "int32")]:
        tokens = output((r,), "int32", position=1)
        with T.Kernel(r, threads=256) as slot:
            tx = T.get_thread_binding()
            best = T.alloc_var("float32")
            index = T.alloc_var("int32")
            values = T.alloc_shared((256,), "float32")
            indices = T.alloc_shared((256,), "int32")
            best = -T.infinity("float32")
            index = vocab
            for tile in T.serial(T.ceildiv(vocab, 256)):
                i = tile * 256 + tx
                if i < vocab:
                    if (logits[slot, i] > best) | (logits[slot, i] == best) & (
                        i < index
                    ):
                        best = logits[slot, i]
                        index = i
            values[tx] = best
            indices[tx] = index
            T.sync_threads()
            for step in T.unroll(8):
                stride = 128 >> step
                if tx < stride:
                    if (values[tx + stride] > values[tx]) | (
                        values[tx + stride] == values[tx]
                    ) & (indices[tx + stride] < indices[tx]):
                        values[tx] = values[tx + stride]
                        indices[tx] = indices[tx + stride]
                T.sync_threads()
            if tx == 0:
                tokens[slot] = T.if_then_else(active[slot] != 0, indices[0], -1)
                if active[slot] != 0:
                    positions[slot] += 1
        return (tokens, positions)

    return compute


def native_embedding(spec: KernelSpec, schedule: Mapping[str, object]):
    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    r, c, vocab = (p["r"], p["c"], p["vocab"])

    @kernel
    def compute(
        ids: Tensor((r,), "int32"), w: Tensor((vocab, c), "bfloat16")
    ) -> Tensor((r, c), "bfloat16"):
        out = output((r, c), "bfloat16")
        with T.Kernel(r, T.ceildiv(c, 256), threads=256) as (b, tile):
            for j in T.Parallel(256):
                col = tile * 256 + j
                if col < c:
                    out[b, col] = T.if_then_else(
                        ids[b] >= 0, w[T.max(ids[b], 0), col], 0
                    )
        return out

    return compute


def argmax(spec: KernelSpec, schedule: Mapping[str, object]):
    p = {**spec.parameters, **schedule}
    import tilelang.language as T
    from akas.kernels.reductions import tree_reduce

    r, c = (p["r"], p["c"])
    threads = p.get("threads", 64)
    if threads not in (64, 128, 256, 512, 1024):
        raise ValueError("invalid argmax thread count")
    maximum = tree_reduce(threads, "max")

    @kernel
    def compute(
        x: Tensor((r * c,), "float32"), enabled: Tensor((r,), "int32")
    ) -> Tensor((r,), "int32"):
        out = output((r,), "int32", position=1)
        with T.Kernel(r, threads=threads) as row:
            tx = T.get_thread_binding()
            scratch = T.alloc_shared((threads,), "float32")
            best = T.alloc_var("float32")
            index = T.alloc_var("int32")
            best = -T.infinity("float32")
            index = c
            for tile in T.serial(T.ceildiv(c, threads)):
                if tile * threads + tx < c:
                    if x[row * c + tile * threads + tx] > best:
                        best = x[row * c + tile * threads + tx]
                        index = tile * threads + tx
            scratch[tx] = best
            T.sync_threads()
            maximum(scratch, tx)
            value = T.alloc_var("float32")
            value = scratch[0]
            T.sync_threads()
            scratch[tx] = T.cast(T.if_then_else(best == value, -index, -c), "float32")
            T.sync_threads()
            maximum(scratch, tx)
            if tx == 0 and enabled[row] != 0:
                out[row] = T.cast(-scratch[0], "int32")
        return out

    return compute


def embedding(spec: KernelSpec, schedule: Mapping[str, object]):
    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    r, c = (p["r"], p["c"])
    v = p["v"]
    start = p.get("start", 0)

    @kernel
    def compute(
        tokens: Tensor((r,), "int32"),
        w: Tensor((v * c,), p.get("weight_dtype", "float32")),
    ) -> Tensor((r * c,), "float32"):
        out = output((r * c,), "float32")
        with T.Kernel(T.ceildiv(c, 64) * r, threads=64) as flat_block:
            block = flat_block % T.ceildiv(c, 64)
            row = flat_block // T.ceildiv(c, 64)
            tx = T.get_thread_binding()
            col = block * 64 + tx
            if col < c and tokens[row] >= start and (tokens[row] < start + v):
                out[row * c + col] = w[(tokens[row] - start) * c + col]
        return out

    return compute


def embedding_packed(spec: KernelSpec, schedule: Mapping[str, object]):
    from akas.quantization.weights import bind

    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    c = p.get("c")
    r = p.get("r", 1)
    q, v = (p["type"], p["v"])
    weights = bind(p, (v, c), provider="cuda")
    if weights.storage.metadata:
        raise ValueError("this packed operation ABI requires embedded metadata")
    count, dtype = (weights.storage.values.shape[0], weights.storage.values.dtype)
    read_weight = weights.read

    @kernel
    def compute(tokens: Tensor((r,), "int32"), w: Tensor((count,), dtype)) -> Tensor(
        (r * c,), "float32"
    ):
        out = output((r * c,), "float32")
        with T.Kernel(T.ceildiv(c, 256), r, threads=256) as (bx, row):
            for lane in T.Parallel(256):
                col = bx * 256 + lane
                if col < c:
                    out[row * c + col] = read_weight((w,), tokens[row], col)
        return out

    return compute


def argmax_packed(spec: KernelSpec, schedule: Mapping[str, object]):
    from akas.ops.vocabulary.kernel import _argmax as argmax_kernel

    p = {**spec.parameters, **schedule}
    return argmax_kernel(p)


def _argmax(p, explicit_unroll=False):
    import tilelang.language as T

    n, threads = (p["n"], 256)

    @kernel
    def compute(
        logits: Tensor((n,), "float32"), pos: Tensor((2,), "int32")
    ) -> tuple[Tensor((1,), "int32"), Tensor((2,), "int32")]:
        token = output((1,), "int32", position=1)
        if explicit_unroll:
            T.func_attr({"tensor.webgpu.loop_unroll": "explicit"})
        with T.Kernel(1, threads=threads):
            tx = T.get_thread_binding()
            best = T.alloc_var("float32")
            index = T.alloc_var("int32")
            values = T.alloc_shared((threads,), "float32")
            indices = T.alloc_shared((threads,), "int32")
            best = -T.infinity("float32")
            index = n
            for tile in T.serial(T.ceildiv(n, threads)):
                i = tile * threads + tx
                if i < n:
                    if (logits[i] > best) | (logits[i] == best) & (i < index):
                        best = logits[i]
                        index = i
            values[tx] = best
            indices[tx] = index
            T.sync_threads()
            for step in T.unroll(8):
                stride = 128 >> step
                if tx < stride:
                    if (values[tx + stride] > values[tx]) | (
                        values[tx + stride] == values[tx]
                    ) & (indices[tx + stride] < indices[tx]):
                        values[tx] = values[tx + stride]
                        indices[tx] = indices[tx + stride]
                T.sync_threads()
            if tx == 0:
                token[0] = indices[0]
                pos[1] = 1
        return (token, pos)

    return compute


def packed_word_argmax(spec: KernelSpec, schedule: Mapping[str, object]):
    p = {**spec.parameters, **schedule}
    r = p.get("r", 1)
    from akas.ops.vocabulary.kernel import _argmax as argmax_kernel

    return argmax_kernel(p, explicit_unroll=True)


def packed_word_embedding(spec: KernelSpec, schedule: Mapping[str, object]):
    from akas.quantization.weights import bind

    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    r = p.get("r", 1)
    c, q, v = (p["c"], p["type"], p["v"])
    weights = bind(p, (v, c), provider="webgpu")
    if weights.storage.metadata:
        raise ValueError("this packed operation ABI requires embedded metadata")
    count, dtype = (weights.storage.values.shape[0], weights.storage.values.dtype)
    decoder = weights.read

    @kernel
    def compute(tokens: Tensor((r,), "int32"), w: Tensor((count,), dtype)) -> Tensor(
        (r * c,), "float32"
    ):
        out = output((r * c,), "float32")
        with T.Kernel(T.ceildiv(c, 256), r, threads=256) as (bx, row):
            for lane in T.Parallel(256):
                col = bx * 256 + lane
                if col < c:
                    out[row * c + col] = decoder((w,), tokens[row], col)
        return out

    return compute


def argmax_partial(spec, schedule):
    import tilelang.language as T
    from akas.kernels.reductions import tree_reduce

    p = {**spec.parameters, **schedule}
    rows, columns, block = p["r"], p["c"], p["block"]
    parts = (columns + block - 1) // block
    maximum = tree_reduce(128, "max")

    @kernel
    def compute(
        x: Tensor((rows * columns,), "float32"),
    ) -> tuple[Tensor((rows, parts), "float32"), Tensor((rows, parts), "int32")]:
        values = output((rows, parts), "float32")
        indices = output((rows, parts), "int32")
        with T.Kernel(parts, rows, threads=128) as (part, row):
            tx = T.get_thread_binding()
            best = T.alloc_var("float32")
            index = T.alloc_var("int32")
            best = -T.infinity("float32")
            index = columns
            scratch = T.alloc_shared((128,), "float32")
            winners = T.alloc_shared((128,), "int32")
            for tile in T.serial(T.ceildiv(block, 128)):
                col = part * block + tile * 128 + tx
                if col < columns and tile * 128 + tx < block:
                    if x[row * columns + col] > best:
                        best = x[row * columns + col]
                        index = col
            scratch[tx] = best
            T.sync_threads()
            maximum(scratch, tx)
            peak = T.alloc_var("float32")
            peak = scratch[0]
            winners[tx] = T.if_then_else(best == peak, -index, -columns)
            T.sync_threads()
            maximum(winners, tx)
            if tx == 0:
                values[row, part] = peak
                indices[row, part] = -winners[0]
        return values, indices

    return compute


def argmax_merge(spec, schedule):
    import tilelang.language as T
    from akas.kernels.reductions import tree_reduce

    p = {**spec.parameters, **schedule}
    rows, columns, block = p["r"], p["c"], p["block"]
    parts = (columns + block - 1) // block
    maximum = tree_reduce(128, "max")

    @kernel
    def compute(
        values: Tensor((rows, parts), "float32"),
        indices: Tensor((rows, parts), "int32"),
        enabled: Tensor((rows,), "int32"),
    ) -> Tensor((rows,), "int32"):
        out = output((rows,), "int32")
        with T.Kernel(rows, threads=128) as row:
            tx = T.get_thread_binding()
            best = T.alloc_var("float32")
            index = T.alloc_var("int32")
            best = -T.infinity("float32")
            index = columns
            scratch = T.alloc_shared((128,), "float32")
            winners = T.alloc_shared((128,), "int32")
            for tile in T.serial(T.ceildiv(parts, 128)):
                part = tile * 128 + tx
                if part < parts:
                    value = values[row, part]
                    candidate = indices[row, part]
                    if value > best or (value == best and candidate < index):
                        best = value
                        index = candidate
            scratch[tx] = best
            T.sync_threads()
            maximum(scratch, tx)
            peak = T.alloc_var("float32")
            peak = scratch[0]
            winners[tx] = T.if_then_else(best == peak, -index, -columns)
            T.sync_threads()
            maximum(winners, tx)
            if tx == 0 and enabled[row] != 0:
                out[row] = -winners[0]
        return out

    return compute


def codebook_residual(spec, schedule):
    """Add a group of offset audio codebook lookups and their channel biases."""
    import tilelang.language as T

    p = {**spec.parameters, **schedule}
    rows, width, groups, vocab = (p[key] for key in ("r", "c", "groups", "vocab"))

    @kernel
    def compute(
        x: Tensor((rows, width), "float32"),
        ids: Tensor((rows, groups), "int32"),
        table: Tensor((groups * vocab * width,), "float32"),
        bias: Tensor((groups * width,), "float32"),
    ) -> Tensor((rows, width), "float32"):
        out = output((rows, width), "float32")
        with T.Kernel(rows, T.ceildiv(width, 128), threads=128) as (row, tile):
            for lane in T.Parallel(128):
                column = tile * 128 + lane
                if column < width:
                    value = T.alloc_var("float32")
                    value = x[row, column]
                    for group in T.unroll(groups):
                        index = ids[row, group] + group * vocab
                        value += (
                            table[index * width + column] + bias[group * width + column]
                        )
                    out[row, column] = value
        return out

    return compute


def codebook_residual_masked(spec, schedule):
    """Add offset audio codebook lookups while skipping negative missing codes."""
    import tilelang.language as T

    p = {**spec.parameters, **schedule}
    rows, width, groups, vocab = (p[key] for key in ("r", "c", "groups", "vocab"))

    @kernel
    def compute(
        x: Tensor((rows, width), "float32"),
        ids: Tensor((rows, groups), "int32"),
        table: Tensor((groups * vocab * width,), "float32"),
        bias: Tensor((groups * width,), "float32"),
    ) -> Tensor((rows, width), "float32"):
        out = output((rows, width), "float32")
        with T.Kernel(rows, T.ceildiv(width, 128), threads=128) as (row, tile):
            for lane in T.Parallel(128):
                column = tile * 128 + lane
                if column < width:
                    value = T.alloc_var("float32")
                    value = x[row, column]
                    for group in T.unroll(groups):
                        code = ids[row, group]
                        if (code >= 0) & (code < vocab):
                            index = code + group * vocab
                            value += (
                                table[index * width + column]
                                + bias[group * width + column]
                            )
                    out[row, column] = value
        return out

    return compute


def zonos3_frame_embed(spec, schedule):
    """Sum BF16 multi-codebook + text embeddings on device with condition-row overrides."""
    import tilelang.language as T

    p = {**spec.parameters, **schedule}
    rows, width, codebooks, code_vocab, text_vocab, cond_rows = (
        p[k] for k in ("r", "c", "codebooks", "code_vocab", "text_vocab", "cond_rows")
    )
    cols = codebooks + 1
    audio_bytes = codebooks * code_vocab * width * 2
    text_bytes = text_vocab * width * 2

    @T.macro
    def load_bf16(buf, elem_idx):
        lo = T.cast(buf[elem_idx * 2], "uint32")
        hi = T.cast(buf[elem_idx * 2 + 1], "uint32")
        return T.reinterpret("float32", (lo | (hi << 8)) << 16)

    @kernel
    def compute(
        frames: Tensor((rows, cols), "int32"),
        audio_w: Tensor((audio_bytes,), "uint8"),
        text_w: Tensor((text_bytes,), "uint8"),
        cond_w: Tensor((cond_rows, width), "float32"),
    ) -> Tensor((rows, width), "float32"):
        out = output((rows, width), "float32")
        with T.Kernel(rows, T.ceildiv(width, 128), threads=128) as (row, tile):
            c0 = frames[row, 0]
            t_id = frames[row, codebooks]
            for lane in T.Parallel(128):
                col = tile * 128 + lane
                if col < width:
                    if c0 < 0:
                        out[row, col] = 0.0
                    elif t_id < 0:
                        cond_idx = T.min(T.max(-1 - t_id, 0), cond_rows - 1)
                        out[row, col] = cond_w[cond_idx, col]
                    else:
                        acc = T.alloc_var("float32")
                        acc = load_bf16(
                            text_w,
                            T.min(t_id, text_vocab - 1) * width + col,
                        )
                        for cb in T.unroll(codebooks):
                            aid = cb * code_vocab + T.min(
                                T.max(frames[row, cb], 0), code_vocab - 1
                            )
                            acc += load_bf16(audio_w, aid * width + col)
                        out[row, col] = acc
        return out

    return compute


def codebook_sample(spec, schedule):
    """Sample delayed multi-codebook rows with softcap, repetition penalty, top-k and top-p."""
    import tilelang.language as T

    p = {**spec.parameters, **schedule}
    slots = p["slots"]
    codebooks = p.get("codebooks", 10)
    vocab = p.get("vocab", 4098)
    eoa = vocab - 2
    pad = vocab - 1
    window = p.get("window", 96)
    rep_cbs = p.get("rep_codebooks", 6)
    top_k = p.get("top_k", 80)
    top_p = p.get("top_p", 0.69)
    penalty = p.get("penalty", 2.4)
    softcap = p.get("softcap", 15.0)
    silence = p.get("silence_frames", 3)
    text_pad = p.get("text_pad", 8196)
    threads = 128
    per_thread = (eoa + 1 + threads - 1) // threads

    @kernel
    def compute(
        projection: Tensor((slots, codebooks * vocab), "float32"),
        history: Tensor((slots, window, codebooks), "int32"),
        steps: Tensor((slots,), "int32"),
        temperatures: Tensor((slots,), "float32"),
        uniforms: Tensor((slots, codebooks), "float32"),
    ) -> tuple[
        Tensor((slots, window, codebooks), "int32"),
        Tensor((slots, codebooks + 1), "int32"),
    ]:
        out = output((slots, codebooks + 1), "int32")
        with T.Kernel(codebooks, slots, threads=threads) as (cb, slot):
            tx = T.get_thread_binding()
            step = steps[slot]
            hist_sh = T.alloc_shared((window,), "int32")
            sh_val = T.alloc_shared((threads,), "float32")
            sh_idx = T.alloc_shared((threads,), "int32")
            top_vals = T.alloc_shared((top_k,), "float32")
            top_ids = T.alloc_shared((top_k,), "int32")
            probs = T.alloc_local((top_k,), "float32")
            vals = T.alloc_local((per_thread,), "float32")

            if step >= 0:
                if tx < window:
                    hist_sh[tx] = history[slot, tx, cb]
                T.sync_threads()

                for m in T.serial(per_thread):
                    tok = m * threads + tx
                    if tok <= eoa:
                        raw = projection[slot, cb * vocab + tok]
                        v = T.alloc_var("float32")
                        v = T.float32(softcap) * T.tanh(raw / T.float32(softcap))
                        if (cb < rep_cbs) & (tok < eoa):
                            seen = T.alloc_var("int32")
                            seen = 0
                            for w_idx in T.serial(window):
                                if (w_idx < step) & (hist_sh[w_idx] == tok):
                                    seen = 1
                            if seen != 0:
                                v = T.if_then_else(
                                    v > 0.0,
                                    v / T.float32(penalty),
                                    v * T.float32(penalty),
                                )
                        vals[m] = v
                    else:
                        vals[m] = T.float32(-1e30)

                for rank in T.serial(top_k):
                    best_v = T.alloc_var("float32")
                    best_m = T.alloc_var("int32")
                    best_i = T.alloc_var("int32")
                    best_v = T.float32(-1e30)
                    best_m = 0
                    best_i = vocab
                    for m in T.serial(per_thread):
                        tok = m * threads + tx
                        if (tok <= eoa) & (
                            (vals[m] > best_v) | ((vals[m] == best_v) & (tok < best_i))
                        ):
                            best_v = vals[m]
                            best_m = m
                            best_i = tok
                    sh_val[tx] = best_v
                    sh_idx[tx] = best_i
                    T.sync_threads()
                    for stride_exp in T.unroll(7):
                        stride = 64 >> stride_exp
                        if tx < stride:
                            other_v = sh_val[tx + stride]
                            other_i = sh_idx[tx + stride]
                            if (other_v > sh_val[tx]) | (
                                (other_v == sh_val[tx]) & (other_i < sh_idx[tx])
                            ):
                                sh_val[tx] = other_v
                                sh_idx[tx] = other_i
                        T.sync_threads()
                    win_id = sh_idx[0]
                    if tx == 0:
                        top_vals[rank] = sh_val[0]
                        top_ids[rank] = win_id
                    if best_i == win_id:
                        vals[best_m] = T.float32(-1e30)
                    T.sync_threads()

                if tx == 0:
                    temp = temperatures[slot]
                    chosen_tok = T.alloc_var("int32")
                    if temp <= 0.0:
                        chosen_tok = top_ids[0]
                    else:
                        peak = top_vals[0] / temp
                        sum_w = T.alloc_var("float32")
                        sum_w = 0.0
                        for r in T.serial(top_k):
                            w_r = T.exp(top_vals[r] / temp - peak)
                            probs[r] = w_r
                            sum_w += w_r
                        cum = T.alloc_var("float32")
                        kept_sum = T.alloc_var("float32")
                        cum = 0.0
                        kept_sum = 0.0
                        for r in T.serial(top_k):
                            w_r = probs[r]
                            if cum > T.float32(top_p) * sum_w:
                                probs[r] = 0.0
                            else:
                                kept_sum += w_r
                            cum += w_r
                        target = uniforms[slot, cb] * kept_sum
                        cdf = T.alloc_var("float32")
                        picked = T.alloc_var("int32")
                        cdf = 0.0
                        picked = -1
                        for r in T.serial(top_k):
                            cdf += probs[r]
                            if (picked < 0) & (target < cdf):
                                picked = top_ids[r]
                        if picked < 0:
                            picked = top_ids[0]
                        chosen_tok = picked
                    delayed_tok = T.if_then_else(cb > silence + step, pad, chosen_tok)
                    out[slot, cb] = delayed_tok
                    history[slot, step % window, cb] = delayed_tok
                    if cb == 0:
                        out[slot, codebooks] = text_pad
            else:
                if tx == 0:
                    out[slot, cb] = pad
                    if cb == 0:
                        out[slot, codebooks] = text_pad
        return history, out

    return compute
