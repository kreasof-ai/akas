"""Webgpu definitions for kernels."""


def subgroup_reduce(size, operation="sum"):
    import tilelang.language as T

    builtin = "subgroupMax" if operation == "max" else "subgroupAdd"

    @T.macro
    def reduce(scratch, tx, value: T.Ref):
        scratch[tx] = T.call_extern("float32", builtin, value)
        T.sync_threads()
        if tx == 0:
            value = -T.infinity("float32") if operation == "max" else 0
            for group in T.serial(
                T.ceildiv(
                    size,
                    T.cast(T.call_extern("uint32", "tensor_subgroup_size"), "int32"),
                )
            ):
                i = group * T.cast(
                    T.call_extern("uint32", "tensor_subgroup_size"), "int32"
                )
                if operation == "max":
                    value = T.max(value, scratch[i])
                else:
                    value = value + scratch[i]
            scratch[0] = value
        T.sync_threads()

    return reduce


def segmented_reduce(lanes):
    import tilelang.language as T

    def shuffles(acc):
        for stride in (lanes >> i for i in range(1, lanes.bit_length())):
            shuffle(acc, stride)

    def steps(scratch, tx, lane):
        for stride in (lanes >> i for i in range(1, lanes.bit_length())):
            step(scratch, tx, lane, stride)

    @T.macro
    def shuffle(acc: T.Ref, stride):
        acc = acc + T.call_extern(
            "float32", "subgroupShuffleXor", acc, T.uint32(stride)
        )

    @T.macro
    def step(scratch, tx, lane, stride):
        if lane < stride:
            scratch[tx] = scratch[tx] + scratch[tx + stride]
        T.sync_threads()

    @T.macro
    def segmented(acc: T.Ref, scratch, tx, lane):
        if T.call_extern("uint32", "tensor_subgroup_size") >= lanes:
            shuffles(acc)
            scratch[tx] = acc
        else:
            scratch[tx] = acc
            T.sync_threads()
            steps(scratch, tx, lane)

    return segmented
