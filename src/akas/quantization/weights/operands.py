"""Packed, exact FP8-to-FP16 operand conversion for paired Hopper MMA."""

CUDA_SOURCE = r"""
__device__ __forceinline__ unsigned int tensor_widen_e4m3x2(unsigned short bits) {
    unsigned int packed;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(packed) : "h"(bits));
    return packed;
}
"""


def widen(p):
    import tilelang.language as T

    shape = tuple(p["shape"])
    if len(shape) != 3:
        raise ValueError("expert operand cache needs [experts,output,input]")
    experts, rows, k = shape
    count = rows * k
    if k % 32:
        raise ValueError("operand pairing requires K32")

    @T.prim_func
    def kernel(source: T.Tensor(shape, "uint8"), out: T.Tensor(shape, "float16")):
        with T.Kernel(experts, T.ceildiv(count, 1024), threads=256) as (expert, block):
            for index in T.Parallel(1024):
                pos = block * 1024 + index
                if pos < count:
                    row = pos // k
                    j = pos % 32
                    original = (
                        (pos % k) // 32 * 32 + (j % 16) // 2 * 4 + j % 2 + j // 16 * 2
                    )
                    out[expert, row, pos % k] = T.reinterpret(
                        "float8_e4m3fn", source[expert, row, original]
                    )

    return kernel
