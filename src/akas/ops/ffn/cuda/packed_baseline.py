"""Operation kernels for ffn.cuda.packed_baseline."""

from akas.ops.ffn.activation import swiglu_kernel


def make_kernel(kind, p):
    if kind == "swiglu":
        return swiglu_kernel(dict(p, flat=True, input_names=("x", "y")))
    raise ValueError("unsupported operation: " + kind)
