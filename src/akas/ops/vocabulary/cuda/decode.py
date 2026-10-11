"""Operation kernels for vocabulary.cuda.decode."""


def make_kernel(kind, p):
    import tilelang.language as T

    if kind == "embedding":
        r, c, vocab = p["r"], p["c"], p["vocab"]

        @T.prim_func
        def kernel(
            ids: T.Tensor((r,), "int32"),
            w: T.Tensor((vocab, c), "bfloat16"),
            out: T.Tensor((r, c), "bfloat16"),
        ):
            with T.Kernel(r, T.ceildiv(c, 256), threads=256) as (b, tile):
                for j in T.Parallel(256):
                    col = tile * 256 + j
                    if col < c:
                        out[b, col] = T.if_then_else(
                            ids[b] >= 0, w[T.max(ids[b], 0), col], 0
                        )

        return kernel
    import tilelang.language as T

    if kind == "argmax":
        r, vocab = p["r"], p["vocab"]

        @T.prim_func
        def kernel(
            logits: T.Tensor((r, vocab), "float32"),
            tokens: T.Tensor((r,), "int32"),
            positions: T.Tensor((r,), "int32"),
            active: T.Tensor((r,), "int32"),
        ):
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
                        if (logits[slot, i] > best) | (
                            (logits[slot, i] == best) & (i < index)
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
                            (values[tx + stride] == values[tx])
                            & (indices[tx + stride] < indices[tx])
                        ):
                            values[tx] = values[tx + stride]
                            indices[tx] = indices[tx + stride]
                    T.sync_threads()
                if tx == 0:
                    tokens[slot] = T.if_then_else(active[slot] != 0, indices[0], -1)
                    if active[slot] != 0:
                        positions[slot] += 1

        return kernel
    raise ValueError("unsupported operation: " + kind)
