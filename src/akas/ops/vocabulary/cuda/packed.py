"""Operation kernels for vocabulary.cuda.packed."""

from akas.ops.vocabulary.cuda.argmax import argmax_kernel


def make_kernel(kind, p):
    if kind == "argmax":
        return argmax_kernel(p)
    raise ValueError("unsupported operation: " + kind)
