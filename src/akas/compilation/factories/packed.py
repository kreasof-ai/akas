"""Producer-only routing; arithmetic lives with each operation."""

from akas.compilation.factories.routing import operation


_PACKED_CUDA = {
    "linear": "linear.cuda.packed",
    "linear_add": "linear.cuda.packed",
    "ffn": "linear.cuda.packed",
    "attention_partial": "attention.cuda.packed",
    "attention_grouped": "attention.cuda.packed",
    "attention_merge": "attention.cuda.packed",
    "qnorm": "normalization.cuda.packed",
    "kvnorm": "normalization.cuda.packed",
    "argmax": "vocabulary.cuda.packed",
    "prefill_tail": "sequence.cuda.packed",
    "add_rms": "normalization.cuda.packed",
}


def packed_cuda_factory(kind, p):
    module = _PACKED_CUDA.get(kind)
    if module is not None:
        return operation(module, kind, p)
    return packed_baseline_factory(kind, p)


_PACKED_BASELINE = {
    "linear": "linear.cuda.packed_baseline",
    "rms": "normalization.cuda.packed_baseline",
    "swiglu": "ffn.cuda.packed_baseline",
    "add": "residual.cuda.packed_baseline",
    "embedding": "vocabulary.cuda.packed_baseline",
    "conv": "convolution.cuda.packed_baseline",
    "qkv": "attention.cuda.packed_baseline",
    "attention": "attention.cuda.packed_baseline",
    "last": "sequence.cuda.packed_baseline",
    "advance": "sequence.cuda.packed_baseline",
}


def packed_baseline_factory(kind, p):
    module = _PACKED_BASELINE.get(kind)
    if module is not None:
        return operation(module, kind, p)
    raise ValueError("unsupported packed_baseline operation: " + kind)


_PACKED_WEBGPU = {
    "linear": "linear.webgpu.packed",
    "ffn": "linear.webgpu.packed",
    "linear_q16": "linear.webgpu.packed",
    "ffn_q16": "linear.webgpu.packed",
    "linear_add": "linear.webgpu.packed",
    "quantize_q8": "quantization.webgpu.packed",
    "quantize_q16": "quantization.webgpu.packed",
    "linear_q8": "linear.webgpu.packed",
    "add_rms": "normalization.webgpu.packed",
    "rms": "normalization.webgpu.packed",
    "qnorm": "normalization.webgpu.packed",
    "kvnorm": "normalization.webgpu.packed",
    "attention_scores": "attention.webgpu.packed",
    "attention": "attention.webgpu.packed",
    "embedding": "vocabulary.webgpu.packed",
    "argmax": "vocabulary.webgpu.packed",
    "prefill_tail": "sequence.webgpu.packed",
    "qkv": "attention.webgpu.packed",
}


def packed_webgpu_factory(kind, p):
    if kind == "rms" and p.get("r", 1) != 1 and not p.get("parallel_rows"):
        return packed_baseline_factory(kind, p)
    module = _PACKED_WEBGPU.get(kind)
    if module is not None:
        return operation(module, kind, p)
    return packed_baseline_factory(kind, p)
