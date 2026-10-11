"""Operation kernels for normalization.cuda.packed_baseline."""

from akas.ops.normalization.kernels import rms_kernel


def make_kernel(kind, p):
    if kind == "rms":
        return rms_kernel(dict(p, flat=True))
    raise ValueError("unsupported operation: " + kind)
