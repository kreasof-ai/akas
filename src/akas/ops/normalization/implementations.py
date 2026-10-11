"""Compatible kernels and search spaces owned by this operation."""

from akas.ops.registry import variants


def parallel_rms(p):
    return p.get("r", 1) == 1 or bool(p.get("parallel_rows"))


IMPLEMENTATIONS = (
    *variants(
        ("qnorm", "kvnorm", "add_rms"),
        "normalization.cuda.packed",
        "packed",
        priority=20,
    ),
    *variants(("rms",), "normalization.cuda.packed_baseline", "packed-baseline"),
    *variants(("rms",), "normalization.cuda.packed_baseline", "packed", priority=0),
    *variants(
        ("rms",),
        "normalization.cuda.packed_baseline",
        "packed",
        provider="webgpu",
        priority=0,
    ),
    *variants(
        ("add_rms", "qnorm", "kvnorm"),
        "normalization.webgpu.packed",
        "packed",
        provider="webgpu",
        priority=20,
    ),
    *variants(
        ("rms",),
        "normalization.webgpu.packed",
        "packed",
        provider="webgpu",
        priority=20,
        accepts=parallel_rms,
    ),
    *variants(
        ("rms", "add_rms"), "normalization.cuda.decode", "block-fp8", phase="decode"
    ),
    *variants(("rms", "add_rms"), "normalization.cuda.decode", "bf16"),
)
