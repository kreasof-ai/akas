"""Contracts and implementation choices for convolution."""

from akas.kernels.contracts import Implementation
from akas.ops.convolution.validation import validation

OUTPUT_ARGUMENTS = {"reference": {"conv": (2, 3)}}
VALIDATIONS = {"conv": validation}
IMPLEMENTATIONS = (
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
        "conv_commit": OutputABI((0,), ("state",)),
    }
)
VALIDATIONS.update(
    {kind: verification_validation for kind in ("conv_verify", "conv_commit")}
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
    for kind, factory in (("conv_verify", "verify"), ("conv_commit", "commit"))
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
            "conv_commit",
            call.parameters,
            state=commit.buffers[fields["state"]],
            journal=commit.buffers[fields["journal"]],
            accepted=commit.buffers[fields["enabled"]],
        )
