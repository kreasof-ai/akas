"""Compatibility entry points; selection belongs to operation registrations."""

from akas.ops.registry import make_kernel


def decode_factory(kind, p):
    return make_kernel(kind, p, contract="block-fp8", provider="cuda", phase="decode")


def prefill_factory(kind, p):
    return make_kernel(kind, p, contract="block-fp8", provider="cuda", phase="prefill")


def verification_factory(kind, p):
    return make_kernel(
        kind, p, contract="block-fp8", provider="cuda", phase="verification"
    )
