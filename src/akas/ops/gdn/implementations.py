"""Compatible kernels and search spaces owned by this operation."""

from akas.ops.registry import variants


IMPLEMENTATIONS = (
    *variants(
        ("gdn_conv", "gdn_prepare", "gdn_scan", "gdn_norm"),
        "gdn.cuda.bf16_batch",
        "bf16",
    ),
    *variants(
        ("gdn_conv", "gdn_prepare", "gdn_norm", "gdn_recurrent"),
        "gdn.cuda.decode",
        "block-fp8",
        phase="decode",
    ),
    *variants(
        ("gdn_conv", "gdn_scan"),
        "gdn.cuda.prefill",
        "block-fp8",
        phase="prefill",
        required=("slots", "chunk"),
    ),
    *variants(
        ("gdn_conv", "gdn_scan", "restore"),
        "gdn.cuda.verification",
        "block-fp8",
        phase="verification",
        required=("slots", "chunk"),
    ),
)
