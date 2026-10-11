"""Operation kernels for quantization.cuda.decode."""


def make_kernel(kind, p):
    import tilelang.language as T

    if kind == "e4m3_decode":

        @T.prim_func
        def kernel(bits: T.Tensor((256,), "uint8"), out: T.Tensor((256,), "float32")):
            with T.Kernel(1, threads=256):
                for i in T.Parallel(256):
                    out[i] = T.call_extern(
                        "float32", "tensor_decode_e4m3", T.cast(bits[i], "uint32")
                    )

        return kernel
    raise ValueError("unsupported operation: " + kind)
