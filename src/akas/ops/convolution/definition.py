"""Contracts and implementation choices for convolution."""

from akas.kernels.contracts import Implementation
from akas.ops.convolution.validation import validation

OUTPUT_ARGUMENTS = {"reference": {"conv": (2, 3)}}
VALIDATIONS = {"conv": validation}


def lower(graph, commit, settings, *, provider, target):
    if (
        set(settings) - {"fused_staging"}
        or type(settings.get("fused_staging", False)) is not bool
    ):
        raise ValueError("unknown convolution staging policy")
    if settings.get("fused_staging"):
        if provider != "cuda" or graph.contract != "reference":
            raise ValueError(
                "convolution staging fusion currently requires CUDA reference arithmetic"
            )
        from akas.compilation.storage import fuse_output_casts

        graph = fuse_output_casts(
            graph, {"conv": 3}, cast_kind="linear_half_input", dtype="float16"
        )
    return graph, commit


IMPLEMENTATIONS = (
    Implementation(
        "conv",
        "akas.ops.convolution.cuda",
        "reference",
        factory="parallel_prefill",
        name="convolution.cuda.parallel_prefill",
        priority=20,
        targets=("sm_89",),
        accepts=lambda p: (
            p.get("chunk", 1) >= 32
            and p["c"] >= 128
            and (p.get("time_tile", p["chunk"]) + 3 * p.get("length", 3) - 2)
            * p.get("columns", 8 if p["chunk"] >= 128 else 16)
            * 4
            <= 90000
        ),
        space=(
            ("columns", (8, 16, 32, 64, 128)),
            ("threads", (256, 128)),
            ("time_tile", (32, 64, 128, 256, 512, 2048)),
        ),
    ),
    Implementation(
        "conv",
        "akas.ops.convolution.kernel",
        "packed-baseline",
        factory="convolution_flat",
        name="convolution.cuda.packed_baseline",
    ),
    Implementation(
        "conv",
        "akas.ops.convolution.kernel",
        "packed",
        priority=0,
        factory="convolution_flat",
        name="convolution.cuda.packed_baseline",
    ),
    Implementation(
        "conv",
        "akas.ops.convolution.kernel",
        "packed",
        provider="webgpu",
        priority=0,
        factory="convolution_flat",
        name="convolution.cuda.packed_baseline",
    ),
    Implementation(
        "conv",
        "akas.ops.convolution.kernel",
        "reference",
        factory="convolution",
        name="convolution.reference",
    ),
    Implementation(
        "conv",
        "akas.ops.convolution.kernel",
        "reference",
        provider="webgpu",
        factory="convolution",
        name="convolution.reference",
    ),
)


from akas.kernels.contracts import OutputABI
from akas.ops.convolution.validation import verification_validation

OUTPUT_ARGUMENTS["reference"].update(
    {
        "conv_verify": OutputABI((3, 4), ("journal", "out")),
        "conv_commit_prefix": OutputABI((0,), ("state",)),
    }
)
VALIDATIONS.update(
    {kind: verification_validation for kind in ("conv_verify", "conv_commit_prefix")}
)
IMPLEMENTATIONS += tuple(
    Implementation(
        kind,
        "akas.ops.convolution.kernel",
        "reference",
        provider=provider,
        factory=factory,
        name=f"convolution.{factory}",
    )
    for provider in ("cuda", "webgpu")
    for kind, factory in (("conv_verify", "verify"), ("conv_commit_prefix", "commit"))
)


def commit_histories(graph, commit):
    """Compose each convolution journal's accepted-prefix commit into a model phase."""
    for call in graph.calls:
        if call.kind != "conv_verify":
            continue
        fields = (
            dict(call.bindings)
            if call.bindings
            else dict(
                zip(("x", "w", "state", "journal", "out", "enabled"), call.arguments)
            )
        )
        commit(
            "conv_commit_prefix",
            call.parameters,
            state=commit.buffers[fields["state"]],
            journal=commit.buffers[fields["journal"]],
            accepted=commit.buffers[fields["enabled"]],
        )
