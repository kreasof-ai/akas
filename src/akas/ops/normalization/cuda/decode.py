"""Operation kernels for normalization.cuda.decode."""

from akas.ops.normalization.kernels import rms_kernel


def make_kernel(kind, p):
    if kind in ("rms", "add_rms"):
        return rms_kernel(
            dict(
                p,
                fused=kind == "add_rms",
                zero_centered=True,
                round_inputs=True,
                input_name="x" if kind == "add_rms" else "residual",
                input_dtype="float32" if kind == "add_rms" else "bfloat16",
                residual_dtype="bfloat16",
                output_dtype="bfloat16",
                weight_dtype="bfloat16",
            )
        )
    raise ValueError("unsupported operation: " + kind)
