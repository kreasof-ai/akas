"""Producer-only routing; arithmetic lives with each operation."""

from akas.profiles.routing import operation


_DECODE = {
    "moe_groups": "moe.cuda.decode",
    "fp8_experts": "moe.cuda.decode",
    "bf16_linear": "linear.cuda.decode",
    "embedding": "vocabulary.cuda.decode",
    "rms": "normalization.cuda.decode",
    "add_rms": "normalization.cuda.decode",
    "router": "moe.cuda.decode",
    "swiglu": "ffn.cuda.decode",
    "swiglu_experts": "ffn.cuda.decode",
    "moe_combine": "moe.cuda.decode",
    "gdn_conv": "gdn.cuda.decode",
    "gdn_prepare": "gdn.cuda.decode",
    "gdn_norm": "gdn.cuda.decode",
    "attention_q": "attention.cuda.decode",
    "attention_kv": "attention.cuda.decode",
    "attention_partial": "attention.cuda.decode",
    "attention_merge": "attention.cuda.decode",
    "argmax": "vocabulary.cuda.decode",
    "e4m3_decode": "quantization.cuda.decode",
    "fp8_linear": "linear.cuda.decode",
    "gdn_recurrent": "gdn.cuda.decode",
}


def decode_factory(kind, p):
    if p.get("kv_dtype") == "fp8":
        from akas.ops.attention.cuda.kv_fp8 import make_kernel

        return make_kernel(kind, p)
    module = _DECODE.get(kind)
    if module is not None:
        return operation(module, kind, p)
    raise ValueError("unsupported decode operation: " + kind)


_PREFILL = {
    "controls": "sequence.cuda.prefill",
    "gdn_conv": "gdn.cuda.prefill",
    "gdn_scan": "gdn.cuda.prefill",
    "expert_routes": "moe.cuda.prefill",
    "last_rows": "sequence.cuda.prefill",
    "advance": "sequence.cuda.prefill",
    "attention_kv": "attention.cuda.prefill",
    "attention": "attention.cuda.prefill",
}


def prefill_factory(kind, p):
    if p.get("kv_dtype") == "fp8":
        from akas.ops.attention.cuda.kv_fp8 import make_kernel

        return make_kernel(kind, p)
    # These contracts require explicit slot and chunk geometry.
    p["slots"], p["chunk"]
    module = _PREFILL.get(kind)
    if module is not None:
        return operation(module, kind, p)
    raise ValueError("unsupported prefill operation: " + kind)


_VERIFICATION = {
    "gdn_conv": "gdn.cuda.verification",
    "gdn_scan": "gdn.cuda.verification",
    "restore": "gdn.cuda.verification",
}


def verification_factory(kind, p):
    # These contracts require explicit slot and chunk geometry.
    p["slots"], p["chunk"]
    module = _VERIFICATION.get(kind)
    if module is not None:
        return operation(module, kind, p)
    raise ValueError("unsupported verification operation: " + kind)
