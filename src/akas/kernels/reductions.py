"""Reductions definitions for kernels."""


def tree_reduce(size, operation="sum"):
    import tilelang.language as T

    def reduce(scratch, tx):
        for stride in (size >> i for i in range(1, size.bit_length())):
            step(scratch, tx, stride)

    @T.macro
    def step(scratch, tx, stride):
        if tx < stride:
            if operation == "max":
                scratch[tx] = T.max(scratch[tx], scratch[tx + stride])
            else:
                scratch[tx] = scratch[tx] + scratch[tx + stride]
        T.sync_threads()

    return reduce
