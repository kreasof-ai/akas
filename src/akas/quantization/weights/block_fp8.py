"""Block E4M3 storage, decoding, and independent host references."""


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



def decode_storage(bits):
    import numpy as np

    bits = np.asarray(bits, dtype="uint8")
    exponent = ((bits >> 3) & 15).astype("int32")
    fraction = (bits & 7).astype("float32")
    magnitude = np.where(exponent == 0, fraction / 512,
                         (1 + fraction / 8) * np.exp2(exponent - 7)).astype("float32")
    return np.where(bits & 128, -magnitude, magnitude).astype("float32")


def encode_storage(values):
    """Nearest-even E4M3 with finite saturation after explicit F32 rounding."""
    import numpy as np

    values = np.asarray(values, dtype="float32")
    bits = values.view("uint32")
    magnitude = np.abs(values)
    rounded = (bits & np.uint32(0x7fffffff)) + np.uint32(0x7ffff) + ((bits >> 20) & 1)
    normal = (rounded >> 20).astype("int32") - 960
    subnormal = np.rint(magnitude * np.float32(512)).astype("int32")
    code = np.clip(np.where(magnitude < np.float32(0.015625), subnormal, normal), 0, 126)
    return code.astype("uint8") | ((bits >> 24) & 128).astype("uint8")


def validation(kind, parameters, signature):
    import numpy as np
    from akas.ops.contracts import ValidationCase, sample_arguments

    values = sample_arguments(signature)
    values["bits"][:] = np.arange(256, dtype="uint8")
    values["bits"][[127, 255]] = [126, 254]
    expected = {name: value.copy() for name, value in values.items()}
    expected["out"][:] = decode_storage(values["bits"])
    return ValidationCase(values, expected, rtol=0, atol=0)


from akas.ops.registry import variants

IMPLEMENTATIONS = variants(("e4m3_decode",), "akas.quantization.weights.block_fp8", "block-fp8")
VALIDATIONS = {"e4m3_decode": validation}
