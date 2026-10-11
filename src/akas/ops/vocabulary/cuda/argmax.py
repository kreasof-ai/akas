"""Packed GGML weights and baseline inference operators.

Factories are producer-only; consumers execute Tensor artifacts."""


def argmax_kernel(p, explicit_unroll=False):
    import tilelang.language as T

    n, threads = (p["n"], 256)

    @T.prim_func
    def kernel(
        logits: T.Tensor((n,), "float32"),
        token: T.Tensor((1,), "int32"),
        pos: T.Tensor((2,), "int32"),
    ):
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

    return kernel
