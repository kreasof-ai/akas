"""Compatibility entry points; selection belongs to operation registrations."""

from akas.ops.registry import make_kernel


def packed_baseline_factory(kind, p):
    return make_kernel(
        kind, p, contract="packed-baseline", provider="cuda", phase="forward"
    )


def packed_cuda_factory(kind, p):
    return make_kernel(kind, p, contract="packed", provider="cuda", phase="forward")


def packed_webgpu_factory(kind, p):
    return make_kernel(kind, p, contract="packed", provider="webgpu", phase="forward")
