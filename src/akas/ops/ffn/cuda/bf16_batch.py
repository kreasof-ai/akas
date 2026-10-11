"""Operation kernels for ffn.cuda.bf16_batch."""

from akas.ops.ffn.activation import swiglu_kernel


def make_kernel(kind, p):
    r = p.get("r", 1)
    slots, chunk, pool = p.get("slots", r), p.get("chunk", 1), p.get("pool", 256)
    rows = slots * chunk
    if kind == "swiglu":
        return swiglu_kernel(
            dict(p, r=rows, packed=True, output_dtype="bfloat16", round_inputs=True)
        )
    raise ValueError("unsupported operation: " + kind)
