"""Compatible kernels and search spaces owned by this operation."""

from akas.ops.registry import variants
from akas.quantization.kv import is_quantized


def quantized_cache(p):
    return is_quantized(p.get("kv_dtype", "bfloat16"))


IMPLEMENTATIONS = (
    *variants(
        ("attention", "attention_kv"),
        "attention.cuda.quantized",
        "block-fp8",
        phase="prefill",
        priority=30,
        accepts=quantized_cache,
    ),
    *variants(
        ("attention_partial", "attention_kv"),
        "attention.cuda.quantized",
        "block-fp8",
        phase="decode",
        priority=30,
        accepts=quantized_cache,
    ),
    *variants(
        ("attention_partial", "attention_grouped", "attention_merge"),
        "attention.cuda.packed",
        "packed",
        priority=20,
    ),
    *variants(
        ("qkv", "attention"), "attention.cuda.packed_baseline", "packed-baseline"
    ),
    *variants(
        ("qkv", "attention"), "attention.cuda.packed_baseline", "packed", priority=0
    ),
    *variants(
        ("qkv", "attention"),
        "attention.cuda.packed_baseline",
        "packed",
        provider="webgpu",
        priority=0,
    ),
    *variants(
        ("attention_scores", "attention", "qkv"),
        "attention.webgpu.packed",
        "packed",
        provider="webgpu",
        priority=20,
    ),
    *variants(("attention_qkv", "attention"), "attention.cuda.bf16_batch", "bf16"),
    *variants(
        ("attention_kv", "attention_partial"),
        "attention.cuda.decode",
        "block-fp8",
        phase="decode",
    ),
    *variants(("attention_q", "attention_merge"), "attention.cuda.decode", "block-fp8"),
    *variants(
        ("attention_kv", "attention"),
        "attention.cuda.prefill",
        "block-fp8",
        phase="prefill",
        required=("slots", "chunk"),
    ),
)

IMPLEMENTATIONS += tuple(
    item
    for provider in ("cuda", "webgpu")
    for item in variants(
        ("rope", "cache_write", "attention"),
        "attention.reference",
        "reference",
        provider=provider,
    )
)

IMPLEMENTATIONS += variants(
    ("attention", "attention_partial", "attention_merge"),
    "attention.cuda.floating",
    "reference",
    priority=30,
    accepts=lambda p: p.get("optimized", False),
    defaults=(("prefill_precision", "float16"),),
)

IMPLEMENTATIONS += variants(
    ("attention_partial", "attention_merge"),
    "attention.floating",
    "reference",
    provider="webgpu",
    priority=30,
    accepts=lambda p: p.get("optimized", False),
)



from akas.ops.attention.validation import validation

VALIDATIONS = {name: validation for name in ("rope", "cache_write", "attention", "attention_partial", "attention_merge", "attention_qkv", "attention_q", "attention_kv")}


def projected_workspace(graph, rows, config, *, prefix="", splits=None, gated=True):
    """Scratch belongs to attention, independent of checkpoint weight storage."""
    heads, kv_heads, depth = config.heads, config.kv_heads, config.head_dim
    specs = {"attq": (rows, (2 if gated else 1) * heads * depth),
             "attk": (rows, kv_heads * depth), "attv": (rows, kv_heads * depth),
             "attq_normal": (rows, heads, depth), "att_out": (rows, heads * depth)}
    if splits is not None:
        specs.update(att_parts=(rows, kv_heads, splits, heads // kv_heads, depth),
                     att_stats=(rows, kv_heads, splits, heads // kv_heads, 2))
    return {name: graph.buffer(prefix + name, shape,
                              "bfloat16" if name in ("attq_normal", "att_out") else "float32")
            for name, shape in specs.items()}


def attend(graph, x, weights, states, controls, *, config, prefix, context):
    """Compose packed Q/K/V projection and mapped causal attention."""
    from akas.ops.linear.operation import project

    rows, width = x.shape
    common = controls["parameters"]
    projected = project(graph,x,weights[prefix+"input.weight"],width=width,name=prefix+"projection")
    query = graph.buffer(prefix+"query",(rows,config.heads,config.head_dim),"bfloat16")
    attended = graph.buffer(prefix+"attended",(rows,config.heads*config.head_dim),"bfloat16")
    parameters = dict(common,heads=config.heads,kv_heads=config.kv_heads,d=config.head_dim,
                      capacity=context,eps=config.epsilon)
    keys, values = states
    graph("attention_qkv",dict(parameters,theta=config.theta),projected,
          weights[prefix+"q_norm.weight"],weights[prefix+"k_norm.weight"],
          controls["mapping"],controls["flat_positions"],controls["active"],keys,values,query)
    graph("attention",parameters,query,keys,values,projected,
          controls["mapping"],controls["positions"],controls["lengths"],attended)
    return project(graph,attended,weights[prefix+"o_proj.weight"],width=width,name=prefix+"output")
