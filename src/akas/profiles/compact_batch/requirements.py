"""Required BF16 specializations from checkpoint geometry."""

from akas.artifacts.kernels import identity
from akas.profiles.compact_batch.parameters import schedule


def requirements(c, slots, chunk, pool, context, *, verify=False, draft=False):
    rows = slots * chunk
    kernels = {}

    def add(kind, **p):
        kernels[identity(kind, p)] = (kind, p)

    def linear(k, o, r=rows):
        p = schedule(r, k, o, c.width)
        add("split_linear", **p)
        add("split_merge", **p)

    common = dict(slots=slots, chunk=chunk, pool=pool)
    add("controls", **common)
    add("embedding", r=rows, c=c.width, vocab=c.vocab)
    for kind in ("rms", "add_rms"):
        add(kind, r=rows, c=c.width, eps=c.epsilon)
    proj = c.qkv_width + c.value_width + 2 * c.value_heads
    linear(c.width, proj)
    state = dict(commit=False) if verify else {}
    add("gdn_conv", **common, **state, channels=c.qkv_width, projection=proj)
    add(
        "gdn_prepare",
        **common,
        heads=c.value_heads,
        key_heads=c.key_heads,
        d=c.key_dim,
        channels=c.qkv_width,
        projection=proj,
    )
    add("gdn_scan", **common, **state, heads=c.value_heads, d=c.key_dim, tile=32)
    if verify:
        add("gdn_conv", **common, channels=c.qkv_width, projection=proj)
        add("gdn_scan", **common, heads=c.value_heads, d=c.key_dim, tile=32)
    add(
        "gdn_norm",
        **common,
        heads=c.value_heads,
        d=c.value_dim,
        channels=c.qkv_width,
        projection=proj,
        eps=c.epsilon,
    )
    linear(c.value_width, c.width)
    attproj = 2 * c.heads * c.head_dim + 2 * c.kv_heads * c.head_dim
    linear(c.width, attproj)
    att = dict(
        common,
        heads=c.heads,
        kv_heads=c.kv_heads,
        d=c.head_dim,
        capacity=context,
        eps=c.epsilon,
    )
    add("attention_qkv", **att, theta=c.theta)
    add("attention", **att)
    linear(c.heads * c.head_dim, c.width)
    linear(c.width, 2 * c.intermediate)
    add("swiglu", **common, c=c.intermediate)
    linear(c.intermediate, c.width)
    add("last_rows", **common, c=c.width)
    head_rows = rows if verify else slots
    linear(c.width, c.vocab, r=head_rows)
    add("argmax", r=head_rows, vocab=c.vocab)
    add("head_linear", **schedule(head_rows, c.width, c.vocab, c.width))
    add("head_argmax", r=head_rows, tiles=(c.vocab + 63) // 64, vocab=c.vocab)
    if draft:
        add("mtp_join", r=rows, c=c.width, eps=c.epsilon)
        add("mtp_cast", r=rows, c=c.width)
        linear(2 * c.width, c.width)
        add("gather_hidden", **common, c=c.width)
        add("store_hidden", **common, c=c.width)
    return kernels
