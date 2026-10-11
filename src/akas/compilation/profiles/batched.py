"""Batched BF16 hybrid execution kernels driven by checkpoint geometry."""

from dataclasses import asdict, replace
from akas.artifacts.kernels import identity
from akas.artifacts.batched import implementation
from akas.compilation.build import build
from akas.compilation.plan import Bundle, Kernel, Source
from akas.kernels.linear.split_linear import schedule

DEPENDENCIES = (
    'tensor.compiler.frontend.source',
    'tensor.compiler.cuda.lowering',
    "akas.kernels.elementwise",
)


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


def _module(kind):
    if kind in ("embedding", "rms", "add_rms", "argmax"):
        return "akas.kernels.hybrid.hybrid"
    if kind in ("head_linear", "head_argmax"):
        return "akas.kernels.linear.head"
    if kind in ("mtp_join", "mtp_cast"):
        return "akas.kernels.speculation.mtp"
    if kind in ("split_linear", "split_merge"):
        return "akas.kernels.linear.split_linear"
    return "akas.kernels.hybrid.hybrid_batch"


def plan(
    checkpoint,
    *,
    slots,
    chunk=1,
    pool=256,
    context=4096,
    target="sm_89",
    verify=False,
    draft=False,
):
    config = checkpoint.config
    if draft:
        config = replace(config, layers=("full_attention",))
    kernels = [
        Kernel(
            key,
            kind,
            parameters,
            Source(_module(kind), "make_kernel", (kind, parameters), DEPENDENCIES),
        )
        for key, (kind, parameters) in requirements(
            config, slots, chunk, pool, context, verify=verify, draft=draft
        ).items()
    ]
    name = f'{"mtp-" if draft else ""}inference-s{slots}-t{chunk}{"-verify" if verify else ""}.json'
    metadata = dict(
        schema="tensor.qwen35-dense.v1",
        config=asdict(config),
        slots=slots,
        chunk=chunk,
        pool=pool,
        context=context,
        target=target,
        verify=verify,
        draft=draft,
        implementation=implementation(),
    )
    return Bundle(metadata, name, kernels)


def produce(checkpoint, out, **options):
    bundle = plan(checkpoint, **options)
    build(bundle, out)
    from pathlib import Path

    return Path(out) / bundle.filename


def control_plan(*, slots, window, width, target="sm_89"):
    kinds = [
        ("spec_setup", {}),
        ("spec_accept", {}),
        ("spec_repair", {"width": width}),
        ("spec_result", {}),
    ]
    for depth in range(1, window - 1):
        kinds.extend(
            [("spec_draft", {"depth": depth}), ("spec_proposal", {"depth": depth})]
        )
    kernels = []
    for kind, extra in kinds:
        parameters = dict(slots=slots, window=window, **extra)
        kernels.append(
            Kernel(
                identity(kind, parameters),
                kind,
                parameters,
                Source(
                    "akas.kernels.speculation.speculation",
                    "make_kernel",
                    (kind, parameters),
                    DEPENDENCIES,
                ),
            )
        )
    return Bundle(
        dict(
            schema="tensor.qwen35-dense-controls.v1",
            target=target,
            slots=slots,
            window=window,
            implementation=implementation(),
        ),
        f"speculation-s{slots}-w{window}.json",
        kernels,
    )


def produce_control(out, *, slots, window, width, target="sm_89"):
    return build(
        control_plan(slots=slots, window=window, width=width, target=target), out
    )


def recurrent_plan(config, *, slots, window, pool=256, target="sm_89"):
    common = dict(slots=slots, window=window, pool=pool)
    projection = config.qkv_width + config.value_width + 2 * config.value_heads
    kinds = [
        ("defer_accept", {}),
        ("defer_clear", {}),
        ("conv_commit", dict(channels=config.qkv_width, projection=projection)),
    ]
    kinds.extend(
        (kind, dict(heads=config.value_heads, d=config.key_dim, tile=32))
        for kind in ("gdn_deferred", "gdn_materialize")
    )
    kernels = []
    for kind, extra in kinds:
        parameters = dict(common, **extra)
        kernels.append(
            Kernel(
                identity(kind, parameters),
                kind,
                parameters,
                Source(
                    "akas.kernels.recurrent.recurrent",
                    "make_kernel",
                    (kind, parameters),
                    DEPENDENCIES,
                ),
            )
        )
    return Bundle(
        dict(
            schema="tensor.qwen35-recurrent-journal.v1",
            target=target,
            **common,
            implementation=implementation(),
        ),
        f"recurrent-s{slots}-w{window}.json",
        kernels,
    )


def produce_recurrent(out, *, config, slots, window, pool=256, target="sm_89"):
    return build(
        recurrent_plan(config, slots=slots, window=window, pool=pool, target=target),
        out,
    )


def produce_bank(
    checkpoint,
    out,
    *,
    slots=(1, 2, 4, 8, 16, 32, 64, 128, 256),
    chunks=(1, 32),
    verify_chunks=(),
    pool=256,
    context=4096,
    target="sm_89",
    mtp=False,
):
    common = dict(pool=pool, context=context, target=target)
    for count in slots:
        for chunk in chunks:
            if chunk > 1 and count > 32:
                continue
            produce(checkpoint, out, slots=count, chunk=chunk, **common)
        for chunk in verify_chunks:
            produce(checkpoint, out, slots=count, chunk=chunk, verify=True, **common)
        if mtp:
            for chunk in sorted({1, 32, *verify_chunks}):
                if chunk == 32 and count > 32:
                    continue
                produce(checkpoint, out, slots=count, chunk=chunk, draft=True, **common)
            for window in verify_chunks:
                produce_control(
                    out,
                    slots=count,
                    window=window,
                    width=checkpoint.config.width,
                    target=target,
                )
                produce_recurrent(
                    out,
                    config=checkpoint.config,
                    slots=count,
                    window=window,
                    pool=pool,
                    target=target,
                )
    return out
