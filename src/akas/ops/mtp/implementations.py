"""Compatible kernels and search spaces owned by this operation."""

from akas.ops.registry import variants


IMPLEMENTATIONS = (*variants(("mtp_join", "mtp_cast"), "mtp.cuda", "bf16"),)
