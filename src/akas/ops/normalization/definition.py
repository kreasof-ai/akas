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


def fuse_residual_normalization(graph):

    def storage(name):
        while graph.buffers[name].alias_of is not None:
            name = graph.buffers[name].alias_of
        return name

    result, index = ([], 0)
    while index < len(graph.calls):
        pair = graph.calls[index : index + 2]
        if len(pair) == 2:
            addition, normalization = pair
            if (
                graph.contract == "reference"
                and addition.kind == "add"
                and (normalization.kind == "rms")
                and (addition.parameters.get("r") == 1)
                and (len(addition.arguments) == len(normalization.arguments) == 3)
                and (addition.arguments[-1] == normalization.arguments[0])
                and (
                    storage(addition.arguments[-1])
                    not in {storage(name) for name in addition.arguments[:2]}
                )
                and (addition.parameters["c"] == normalization.parameters["c"])
                and all(
                    (
                        len(graph.buffers[name].shape) == 1
                        and graph.buffers[name].dtype == "float32"
                        for name in (*addition.arguments, *normalization.arguments)
                    )
                )
            ):
                result.append(
                    replace(
                        normalization,
                        kind="add_rms",
                        arguments=(*addition.arguments, *normalization.arguments[1:]),
                    )
                )
                index += 2
                continue
        result.append(graph.calls[index])
        index += 1
    graph.calls = result
    return graph


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
        kind, "akas.ops.normalization.kernel", "reference", provider="webgpu",
        priority=-1, defaults=(("algorithm", "parallel-tree"), ("sg", True)),
        space=(("sg", (True, False)),), factory=kind,
        semantic_keys=("r", "c", "eps"),
        accepts=lambda p: p["c"] >= 256 and p["c"] % 256 == 0,
        variant="parallel-subgroup", name="normalization.webgpu.parallel-subgroup",
    )
    for kind in ("rms", "add_rms")
)
