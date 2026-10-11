"""Compatibility entry points; selection belongs to operation registrations."""

from akas.ops.registry import make_kernel


def batch_factory(kind, p):
    return make_kernel(kind, p, contract="bf16", provider="cuda", phase="forward")
