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


# Output locations are part of the operation ABI, independent of compiler passes.
OUTPUT_ARGUMENTS = {"reference": {'rope': (1,), 'cache_write': (2, 3, 4, 5), 'attention': (5,), 'attention_partial': (5,), 'attention_merge': (1,)}}

from akas.ops.contracts import Implementation
from akas.ops.attention.lowering import lower
from akas.ops.attention.pipeline import validation as pipeline_validation

IMPLEMENTATIONS += (
    Implementation("attention_decode_workspace", "attention.cuda.workspace", "block-fp8",
                   factory="decode", parameters_only=True),
    Implementation("attention_chunk_active", "attention.pipeline", "block-fp8",
                   factory="active", parameters_only=True),
    Implementation("attention_chunk_partial", "attention.cuda.split", "block-fp8",
                   factory="partial", parameters_only=True,
                   accepts=lambda p: not p.get("hopper"),
                   semantic_keys=("slots", "chunk", "capacity", "splits"),
                   space=(("key_rows", (64, 32, 128)), ("packed_loads", (False, True)))),
    Implementation("attention_chunk_partial", "attention.hardware.hopper.attention", "block-fp8",
                   targets=("sm_90a",), factory="partial", parameters_only=True,
                   accepts=lambda p: p.get("hopper", False),
                   semantic_keys=("slots", "chunk", "capacity", "splits"),
                   space=(("query_tokens", (8, 4, 16)), ("key_rows", (64, 32, 128)))),
    Implementation("attention_chunk_merge", "attention.cuda.split", "block-fp8",
                   factory="merge", parameters_only=True),
    Implementation("attention_chunk_hopper", "attention.hardware.hopper.prefill", "block-fp8",
                   targets=("sm_90a",), factory="attention", parameters_only=True,
                   semantic_keys=("slots", "chunk", "capacity"),
                   space=(("query_rows", (128, 64, 256)), ("threads", (256, 128, 512)))),
)
VALIDATIONS.update({name: pipeline_validation for name in (
    "attention_decode_workspace", "attention_chunk_active", "attention_chunk_partial",
    "attention_chunk_merge", "attention_chunk_hopper")})

OUTPUT_ARGUMENTS["block-fp8"] = {
    "attention_decode_workspace": (-2, -1),
    "attention_chunk_active": (-1,),
    "attention_chunk_partial": (-2, -1),
    "attention_chunk_merge": (-1,),
    "attention_chunk_hopper": (-1,),
}

from akas.ops.contracts import OutputABI

for _contract in ("bf16", "block-fp8"):
    OUTPUT_ARGUMENTS.setdefault(_contract, {}).update({
        "attention_chunk_active": OutputABI((-1,), ("out",)),
        "attention_chunk_merge": OutputABI((-1,), ("out",)),
        "attention_chunk_hopper": OutputABI((-1,), ("out",)),
        "attention_chunk_partial": OutputABI((-2, -1), ("out", "stats")),
        "attention_decode_workspace": OutputABI((-2, -1), ("decoded_k", "decoded_v")),
    })
OUTPUT_ARGUMENTS["bf16"].update({
    "attention_qkv": OutputABI((6, 7, 8), ("kc", "vc", "q")),
    "attention": OutputABI((-1,), ("out",)),
})
OUTPUT_ARGUMENTS["block-fp8"].update({
    "attention_q": OutputABI((-1,), ("q",)),
    "attention_partial": OutputABI((-2, -1), ("out", "stats")),
    "attention_merge": OutputABI((-1,), ("out",)),
    "attention": OutputABI((-1,), ("out",)),
})
