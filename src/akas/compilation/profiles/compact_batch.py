"""Batched BF16 hybrid execution kernels driven by checkpoint geometry."""

from dataclasses import asdict, replace
from akas.artifacts.kernels import identity
from akas.artifacts.fingerprints import compact_batch_hashes as implementation
from akas.compilation.build import build
from akas.compilation.plan import Bundle
from akas.compilation.plan import Kernel
from akas.compilation.plan import Source
from akas.profiles.compact_batch.parameters import schedule

from akas.profiles.compact_batch.requirements import requirements

DEPENDENCIES = (
    "tensor.compiler.frontend.source",
    "tensor.compiler.cuda.lowering",
    "akas.ops.normalization.kernels",
)


def _factory(kind):
    if kind in ("embedding", "rms", "add_rms", "argmax"):
        return "akas.profiles.fixed_slots.factories", "decode_factory"
    if kind in ("head_linear", "head_argmax"):
        return "akas.ops.vocabulary.cuda.fused_head", "make_kernel"
    if kind in ("mtp_join", "mtp_cast"):
        return "akas.ops.mtp.cuda", "make_kernel"
    if kind in ("split_linear", "split_merge"):
        return "akas.ops.linear.cuda.split", "make_kernel"
    return "akas.profiles.compact_batch.factories", "batch_factory"


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
            Source(*_factory(kind), (kind, parameters), DEPENDENCIES),
        )
        for key, (kind, parameters) in requirements(
            config, slots, chunk, pool, context, verify=verify, draft=draft
        ).items()
    ]
    name = f"{'mtp-' if draft else ''}inference-s{slots}-t{chunk}{'-verify' if verify else ''}.json"
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
                    "akas.ops.speculation.cuda",
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
                    "akas.ops.gdn.cuda.journal",
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
