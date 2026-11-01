"""Contracts and implementation choices for normalization."""

from akas.kernels.contracts import OutputABI
from dataclasses import replace
from akas.kernels.contracts import Implementation
from akas.ops.normalization.validation import VALIDATIONS as VALIDATIONS


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


def normalize(
    graph, x, weight, *, epsilon, name, residual=None, role="scratch", dtype="bfloat16"
):
    """Compose ordinary RMS or residual fusion with explicit buffer ownership."""
    rows, width = x.shape
    out = graph.buffer(name, (rows, width), dtype, role=role)
    parameters = dict(r=rows, c=width, eps=epsilon)
    if residual is None:
        graph("rms", parameters, residual=x, w=weight, out=out)
    else:
        graph("add_rms", parameters, x=x, residual=residual, w=weight, out=out)
    return out


def fuse_residual_normalization(graph, *, batched=False):

    def storage(name):
        while graph.buffers[name].alias_of is not None:
            name = graph.buffers[name].alias_of
        return name

    def compatible(addition, normalization):
        if len(addition.arguments) != 3 or len(normalization.arguments) != 3:
            return False
        x, y, residual = (graph.buffers[name] for name in addition.arguments)
        source, weight, out = (graph.buffers[name] for name in normalization.arguments)
        if any(
            len(value.shape) != 1
            or value.dtype not in ("float32", "float16", "bfloat16")
            for value in (x, y, residual, source, weight, out)
        ):
            return False
        a, n = addition.parameters, normalization.parameters
        return (
            a.get("input_dtype", "float32") == x.dtype
            and a.get("other_dtype", x.dtype) == y.dtype
            and a.get("output_dtype", x.dtype) == residual.dtype
            and n.get("input_dtype", "float32") == source.dtype
            and n.get("weight_dtype", "float32") == weight.dtype
            and n.get("output_dtype", source.dtype) == out.dtype
        )

    result, index = ([], 0)
    while index < len(graph.calls):
        pair = graph.calls[index : index + 2]
        if len(pair) == 2:
            addition, normalization = pair
            if (
                graph.contract == "reference"
                and addition.kind == "add"
                and (normalization.kind == "rms")
                and (batched or addition.parameters.get("r") == 1)
                and addition.parameters.get("r", 1)
                == normalization.parameters.get("r", 1)
                and (len(addition.arguments) == len(normalization.arguments) == 3)
                and (addition.arguments[-1] == normalization.arguments[0])
                and (
                    storage(addition.arguments[-1])
                    not in {storage(name) for name in addition.arguments[:2]}
                )
                and (addition.parameters["c"] == normalization.parameters["c"])
                and compatible(addition, normalization)
            ):
                parameters = dict(normalization.parameters)
                typed = any(
                    graph.buffers[name].dtype != "float32"
                    for name in (*addition.arguments, *normalization.arguments)
                )
                if typed:
                    parameters.update(
                        input_dtype=graph.buffers[addition.arguments[0]].dtype,
                        mixed_dtype=graph.buffers[addition.arguments[1]].dtype,
                        residual_dtype=graph.buffers[addition.arguments[2]].dtype,
                        weight_dtype=graph.buffers[normalization.arguments[1]].dtype,
                        output_dtype=graph.buffers[normalization.arguments[2]].dtype,
                    )
                if batched or typed:
                    parameters.update(algorithm="serial-tree", stage_values=True)
                result.append(
                    replace(
                        normalization,
                        kind="add_rms",
                        parameters=parameters,
                        arguments=(*addition.arguments, *normalization.arguments[1:]),
                    )
                )
                index += 2
                continue
        result.append(graph.calls[index])
        index += 1
    graph.calls = result
    return graph


def lower(graph, commit, settings, *, provider, target):
    """Reuse residual normalization for explicitly selected batched pipelines."""
    if set(settings) - {"fused_residual", "fused_staging"}:
        raise ValueError("unknown normalization policy")
    if type(settings.get("fused_residual", False)) is not bool:
        raise ValueError("residual normalization fusion must be a boolean")
    if type(settings.get("fused_staging", False)) is not bool:
        raise ValueError("normalization staging fusion must be a boolean")
    if settings.get("fused_residual"):
        if provider != "cuda" or graph.contract != "reference":
            raise ValueError(
                "batched residual normalization currently requires CUDA reference arithmetic"
            )
        graph = fuse_residual_normalization(graph, batched=True)
    if settings.get("fused_staging"):
        if provider != "cuda" or graph.contract != "reference":
            raise ValueError(
                "normalization staging fusion currently requires CUDA reference arithmetic"
            )
        from akas.compilation.storage import fuse_output_casts

        graph = fuse_output_casts(
            graph,
            {"rms": -1, "add_rms": -1},
            cast_kind="linear_half_input",
            dtype="float16",
        )
    return graph, commit


def search_policies(graph, commit, *, provider, target):
    if provider != "cuda" or graph.contract != "reference":
        return []
    from akas.graph import Graph

    candidate = fuse_residual_normalization(
        Graph.from_metadata(graph.metadata()), batched=True
    )
    return [{"fused_residual": True}] if len(candidate.calls) < len(graph.calls) else []


ROW_INDEPENDENT = ("rms",)
REWRITES = (fuse_residual_normalization,)
OUTPUT_ARGUMENTS = {"reference": {"rms": (-1,), "add_rms": (2, -1)}}
for _contract in ("reference", "packed", "packed-baseline", "bf16", "block-fp8"):
    OUTPUT_ARGUMENTS.setdefault(_contract, {}).update(
        {"rms": OutputABI((-1,), ("out",))}
    )
for _contract in ("bf16", "block-fp8"):
    OUTPUT_ARGUMENTS[_contract]["add_rms"] = OutputABI((1, -1), ("residual", "out"))
for _contract in ("reference", "packed"):
    OUTPUT_ARGUMENTS[_contract]["add_rms"] = OutputABI((2, -1), ("residual", "out"))
IMPLEMENTATIONS = (
    Implementation(
        "qnorm",
        "akas.ops.normalization.kernel",
        "packed",
        priority=20,
        factory="qnorm",
        name="normalization.kernel",
    ),
    Implementation(
        "kvnorm",
        "akas.ops.normalization.kernel",
        "packed",
        priority=20,
        factory="kvnorm",
        name="normalization.kernel",
    ),
    Implementation(
        "add_rms",
        "akas.ops.normalization.kernel",
        "packed",
        priority=20,
        defaults=(("flat", True), ("separate_residual", True)),
        factory="add_rms",
        name="normalization.kernel",
    ),
    Implementation(
        "rms",
        "akas.ops.normalization.kernel",
        "packed-baseline",
        defaults=(("flat", True),),
        factory="rms",
        name="normalization.kernel",
    ),
    Implementation(
        "rms",
        "akas.ops.normalization.kernel",
        "packed",
        priority=0,
        defaults=(("flat", True),),
        variant="fragment",
        factory="rms",
        name="normalization.kernel:fragment",
    ),
    Implementation(
        "rms",
        "akas.ops.normalization.kernel",
        "packed",
        provider="webgpu",
        priority=0,
        defaults=(("flat", True),),
        variant="fragment",
        factory="rms",
        name="normalization.kernel:fragment",
    ),
    Implementation(
        "qnorm",
        "akas.ops.normalization.kernel",
        "packed",
        provider="webgpu",
        priority=20,
        factory="qnorm",
        name="normalization.kernel",
    ),
    Implementation(
        "kvnorm",
        "akas.ops.normalization.kernel",
        "packed",
        provider="webgpu",
        priority=20,
        factory="kvnorm",
        name="normalization.kernel",
    ),
    Implementation(
        "add_rms",
        "akas.ops.normalization.kernel",
        "packed",
        provider="webgpu",
        priority=20,
        defaults=(("algorithm", "parallel-tree"),),
        variant="parallel",
        factory="add_rms",
        name="normalization.kernel:parallel",
    ),
    Implementation(
        "rms",
        "akas.ops.normalization.kernel",
        "packed",
        provider="webgpu",
        priority=20,
        defaults=(("algorithm", "parallel-tree"),),
        variant="parallel",
        accepts=lambda p: p.get("r", 1) == 1 or bool(p.get("parallel_rows")),
        factory="rms",
        name="normalization.kernel:parallel",
    ),
    Implementation(
        "rms",
        "akas.ops.normalization.kernel",
        "bf16",
        defaults=(
            ("zero_centered", True),
            ("round_inputs", True),
            ("input_name", "residual"),
            ("input_dtype", "bfloat16"),
            ("residual_dtype", "bfloat16"),
            ("output_dtype", "bfloat16"),
            ("weight_dtype", "bfloat16"),
        ),
        factory="rms",
        name="normalization.kernel",
    ),
    Implementation(
        "add_rms",
        "akas.ops.normalization.kernel",
        "bf16",
        defaults=(
            ("zero_centered", True),
            ("round_inputs", True),
            ("input_name", "x"),
            ("input_dtype", "float32"),
            ("residual_dtype", "bfloat16"),
            ("output_dtype", "bfloat16"),
            ("weight_dtype", "bfloat16"),
        ),
        factory="add_rms",
        name="normalization.kernel",
    ),
    Implementation(
        "rms",
        "akas.ops.normalization.kernel",
        "block-fp8",
        defaults=(
            ("zero_centered", True),
            ("round_inputs", True),
            ("input_name", "residual"),
            ("input_dtype", "bfloat16"),
            ("residual_dtype", "bfloat16"),
            ("output_dtype", "bfloat16"),
            ("weight_dtype", "bfloat16"),
        ),
        factory="rms",
        name="normalization.kernel",
    ),
    Implementation(
        "add_rms",
        "akas.ops.normalization.kernel",
        "block-fp8",
        defaults=(
            ("zero_centered", True),
            ("round_inputs", True),
            ("input_name", "x"),
            ("input_dtype", "float32"),
            ("residual_dtype", "bfloat16"),
            ("output_dtype", "bfloat16"),
            ("weight_dtype", "bfloat16"),
        ),
        factory="add_rms",
        name="normalization.kernel",
    ),
    Implementation(
        "rms",
        "akas.ops.normalization.kernel",
        "reference",
        defaults=(("algorithm", "serial-tree"),),
        factory="rms",
        name="normalization.kernel",
    ),
    Implementation(
        "rms",
        "akas.ops.normalization.kernel",
        "reference",
        provider="webgpu",
        defaults=(("algorithm", "serial-tree"),),
        factory="rms",
        name="normalization.kernel",
    ),
    Implementation(
        "add_rms",
        "akas.ops.normalization.kernel",
        "reference",
        defaults=(("flat", True), ("separate_residual", True)),
        factory="add_rms",
        name="normalization.kernel",
    ),
    Implementation(
        "add_rms",
        "akas.ops.normalization.kernel",
        "reference",
        provider="webgpu",
        defaults=(("algorithm", "parallel-tree"),),
        factory="add_rms",
        name="normalization.kernel",
    ),
    Implementation(
        "rms",
        "akas.ops.normalization.kernel",
        "bf16",
        provider="webgpu",
        defaults=(
            ("zero_centered", True),
            ("round_inputs", True),
            ("input_name", "residual"),
            ("input_dtype", "float16"),
            ("residual_dtype", "float16"),
            ("output_dtype", "float16"),
            ("weight_dtype", "float16"),
        ),
        factory="rms",
        name="normalization.kernel",
    ),
    Implementation(
        "add_rms",
        "akas.ops.normalization.kernel",
        "bf16",
        provider="webgpu",
        defaults=(
            ("zero_centered", True),
            ("round_inputs", True),
            ("input_name", "x"),
            ("input_dtype", "float32"),
            ("residual_dtype", "float16"),
            ("output_dtype", "float16"),
            ("weight_dtype", "float16"),
        ),
        factory="add_rms",
        name="normalization.kernel",
    ),
)

# The archived RX 6700 XT producer used parallel subgroup RMS for model-width
# rows. Keep it opt-in and independently qualify reduction reassociation.
IMPLEMENTATIONS += tuple(
    Implementation(
        kind,
        "akas.ops.normalization.kernel",
        "reference",
        provider="webgpu",
        priority=-1,
        defaults=(("algorithm", "parallel-tree"), ("sg", True)),
        space=(("sg", (True, False)),),
        factory=kind,
        semantic_keys=("r", "c", "eps"),
        accepts=lambda p: p["c"] >= 64 and p["c"] % 64 == 0,
        variant="parallel-subgroup",
        name="normalization.webgpu.parallel-subgroup",
    )
    for kind in ("rms", "add_rms")
)

OUTPUT_ARGUMENTS["reference"]["post_add_rms"] = OutputABI((-2, -1), ("hidden", "out"))
IMPLEMENTATIONS += (
    Implementation(
        "post_add_rms",
        "akas.ops.normalization.cuda",
        "reference",
        factory="post_add_rms",
        accepts=lambda p: p["c"] % 32 == 0,
    ),
    Implementation(
        "post_add_rms",
        "akas.ops.normalization.webgpu",
        "reference",
        provider="webgpu",
        factory="post_add_rms",
        name="normalization.webgpu.post_add_rms",
        space=(("threads", (256, 128, 64, 32, 512, 1024)),),
        accepts=lambda p: p["c"] % min(256, p["c"]) == 0,
    ),
)
