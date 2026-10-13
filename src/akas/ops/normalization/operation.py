"""Compatible kernels and search spaces owned by this operation."""

from akas.ops.registry import variants


def parallel_rms(p):
    return p.get("r", 1) == 1 or bool(p.get("parallel_rows"))


IMPLEMENTATIONS = (
    *variants(("qnorm", "kvnorm"), "normalization.kernel", "packed", priority=20),
    *variants(
        ("add_rms",),
        "normalization.kernel",
        "packed",
        priority=20,
        defaults=(("flat", True), ("separate_residual", True)),
    ),
    *variants(
        ("rms",), "normalization.kernel", "packed-baseline", defaults=(("flat", True),)
    ),
    *variants(
        ("rms",),
        "normalization.kernel",
        "packed",
        priority=0,
        defaults=(("flat", True),),
        variant="fragment",
    ),
    *variants(
        ("rms",),
        "normalization.kernel",
        "packed",
        provider="webgpu",
        priority=0,
        defaults=(("flat", True),),
        variant="fragment",
    ),
    *variants(
        ("qnorm", "kvnorm"),
        "normalization.kernel",
        "packed",
        provider="webgpu",
        priority=20,
    ),
    *variants(
        ("add_rms",),
        "normalization.kernel",
        "packed",
        provider="webgpu",
        priority=20,
        defaults=(("algorithm", "parallel-tree"),),
        variant="parallel",
    ),
    *variants(
        ("rms",),
        "normalization.kernel",
        "packed",
        provider="webgpu",
        priority=20,
        accepts=parallel_rms,
        defaults=(("algorithm", "parallel-tree"),),
        variant="parallel",
    ),
)

IMPLEMENTATIONS += tuple(
    item
    for contract in ("bf16", "block-fp8")
    for kind in ("rms", "add_rms")
    for item in variants(
        (kind,),
        "normalization.kernel",
        contract,
        phase="forward",
        defaults=(
            ("zero_centered", True),
            ("round_inputs", True),
            ("input_name", "x" if kind == "add_rms" else "residual"),
            ("input_dtype", "float32" if kind == "add_rms" else "bfloat16"),
            ("residual_dtype", "bfloat16"),
            ("output_dtype", "bfloat16"),
            ("weight_dtype", "bfloat16"),
        ),
    )
)

IMPLEMENTATIONS += tuple(
    item
    for provider in ("cuda", "webgpu")
    for item in variants(
        ("rms",),
        "normalization.kernel",
        "reference",
        provider=provider,
        defaults=(("algorithm", "serial-tree"),),
    )
)


from akas.ops.normalization.validation import VALIDATIONS

"""Row independence for graph pruning; kernel arithmetic is unchanged."""

ROW_INDEPENDENT = ("rms",)


def rms_norm(
    emit,
    x,
    weight,
    out,
    *,
    rows,
    width,
    epsilon,
    residual=None,
    input_name="x",
    label=None,
):
    parameters = dict(r=rows, c=width, eps=epsilon)
    if residual is None:
        bindings = {input_name: x, "w": weight, "out": out}
        kind = "rms"
    else:
        bindings = dict(x=x, residual=residual, w=weight, out=out)
        kind = "add_rms"
    emit(kind, parameters, label=label, **bindings)


def normalize(graph, x, weight, *, epsilon, name, residual=None, role="scratch"):
    """Compose ordinary RMS or residual fusion with explicit buffer ownership."""
    rows, width = x.shape
    out = graph.buffer(name, (rows, width), "bfloat16", role=role)
    parameters = dict(r=rows, c=width, eps=epsilon)
    if residual is None:
        graph("rms", parameters, residual=x, w=weight, out=out)
    else:
        graph("add_rms", parameters, x=x, residual=residual, w=weight, out=out)
    return out


IMPLEMENTATIONS += variants(
    ("add_rms",), "normalization.kernel", "reference",
    defaults=(("flat", True), ("separate_residual", True)),
)
IMPLEMENTATIONS += variants(
    ("add_rms",), "normalization.kernel", "reference", provider="webgpu",
    defaults=(("algorithm", "parallel-tree"),),
)

from akas.ops.normalization.rewrites import fuse_residual_normalization

REWRITES = (fuse_residual_normalization,)


# Output locations are part of the operation ABI, independent of compiler passes.
OUTPUT_ARGUMENTS = {"reference": {'rms': (-1,), 'add_rms': (2, -1)}}


from akas.ops.contracts import OutputABI

for _contract in ("reference", "packed", "packed-baseline", "bf16", "block-fp8"):
    OUTPUT_ARGUMENTS.setdefault(_contract, {}).update({'rms': OutputABI((-1,), ('out',))})

for _contract in ("bf16", "block-fp8"):
    OUTPUT_ARGUMENTS[_contract]["add_rms"] = OutputABI((1, -1), ("residual", "out"))
for _contract in ("reference", "packed"):
    OUTPUT_ARGUMENTS[_contract]["add_rms"] = OutputABI((2, -1), ("residual", "out"))
