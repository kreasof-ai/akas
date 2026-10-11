"""Producer-only routing; arithmetic lives with each operation."""

from akas.compilation.factories.routing import operation


_BATCH = {
    "linear": "linear.cuda.bf16_batch",
    "gemv": "linear.cuda.bf16_batch",
    "gather_hidden": "sequence.cuda.bf16_batch",
    "store_hidden": "sequence.cuda.bf16_batch",
    "controls": "sequence.cuda.bf16_batch",
    "gdn_conv": "gdn.cuda.bf16_batch",
    "gdn_prepare": "gdn.cuda.bf16_batch",
    "gdn_scan": "gdn.cuda.bf16_batch",
    "gdn_norm": "gdn.cuda.bf16_batch",
    "swiglu": "ffn.cuda.bf16_batch",
    "last_rows": "sequence.cuda.bf16_batch",
    "attention_qkv": "attention.cuda.bf16_batch",
    "attention": "attention.cuda.bf16_batch",
}


def batch_factory(kind, p):
    module = _BATCH.get(kind)
    if module is not None:
        return operation(module, kind, p)
    raise ValueError("unsupported batch operation: " + kind)
