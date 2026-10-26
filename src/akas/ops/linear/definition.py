"""Contracts and implementation choices for linear."""

from akas.kernels.contracts import OutputABI
from collections import Counter
from dataclasses import replace
from akas.kernels.contracts import Implementation
from akas.ops.linear.validation import pipeline_validation as pipeline_validation
from akas.ops.linear.validation import VALIDATIONS


def search_policies(graph, commit, *, provider, target):
    if (
        graph.contract != "block-fp8"
        or provider != "cuda"
        or not any(call.kind == "fp8_linear" for call in graph.calls)
    ):
        return []
    choices = [{"partitions": count} for count in (2, 4, 8)]
    if target == "sm_90a":
        choices.append({"hopper": True, "partitions": 2})
    return choices


def lower(graph, commit, settings, *, provider, target):
    from akas.graph import Graph
    from akas.quantization import get

    allowed = {
        "partitions",
        "columns",
        "block_m",
        "threads",
        "stages",
        "packed_copy",
        "hopper",
        "packed_gather",
        "mma_reduction",
        "mma_reorder",
        "async_mma",
        "packed_widen",
        "split_ffn",
        "split_ffn_min_rows",
        "stage_weights",
        "stage_inputs",
    }
    if set(settings) - allowed:
        raise ValueError("unknown projection pipeline policy")
    if graph.contract == "reference":
        if set(settings) - {"split_ffn", "split_ffn_min_rows", "stage_weights", "stage_inputs"}:
            raise ValueError("unknown floating projection policy")
        if type(settings.get("split_ffn", False)) is not bool:
            raise ValueError("split FFN policy must be a boolean")
        minimum_rows = settings.get("split_ffn_min_rows", 2)
        if type(minimum_rows) is not int or minimum_rows < 2:
            raise ValueError("split FFN minimum rows must be an integer >= 2")
        if type(settings.get("stage_weights", False)) is not bool:
            raise ValueError("weight staging policy must be a boolean")
        if settings.get("stage_weights") and provider != "webgpu":
            raise ValueError("floating weight staging currently requires WebGPU")
        if type(settings.get("stage_inputs", False)) is not bool:
            raise ValueError("input staging policy must be a boolean")
        if settings.get("stage_inputs") and provider != "webgpu":
            raise ValueError("floating input staging currently requires WebGPU")
        if settings.get("split_ffn"):
            from akas.graph import Call
            calls = []
            for index, call in enumerate(graph.calls):
                p = call.parameters
                if call.kind != "ffn" or p["r"] < minimum_rows or not p.get("ggml"):
                    calls.append(call)
                    continue
                if call.bindings:
                    x, w, w2, out = (call.bindings[n] for n in ("x", "w", "w2", "out"))
                else:
                    x, w, w2, out = call.arguments
                gate = graph.buffer(f"pipeline.gate.{index}", (p["r"] * p["o"],), "float32")
                up = graph.buffer(f"pipeline.up.{index}", (p["r"] * p["o"],), "float32")
                calls += [Call("linear", dict(p), (x, w, gate.name)),
                          Call("linear", dict(p), (x, w2, up.name)),
                          Call("swiglu", dict(r=p["r"], c=p["o"]),
                               (gate.name, up.name, out), label=call.label)]
            graph.calls = calls
        if settings.get("stage_weights"):
            from akas.graph import Call
            calls = []
            for index, call in enumerate(graph.calls):
                p = call.parameters
                if (call.kind not in ("linear", "linear_add", "ffn") or p["r"] < 128
                        or not p.get("ggml") or p.get("q16") or p.get("type") not in (2, 14)):
                    calls.append(call)
                    continue
                if call.bindings:
                    raise ValueError("staged packed projections require positional bindings")
                arguments = list(call.arguments)
                for argument in (1, 2) if call.kind == "ffn" else (1,):
                    staged = graph.buffer(f"pipeline.weight.{index}.{argument}",
                                          (p["o"], p["k"]), "float16")
                    flat = graph.view(staged.name + ".flat", staged, (p["o"] * p["k"],))
                    calls.append(Call("dequantize_ggml", dict(r=p["o"], k=p["k"], type=p["type"],
                        output_dtype="float16", **(dict(weight_layout=p["weight_layout"])
                                                   if "weight_layout" in p else {})),
                        (arguments[argument], staged.name)))
                    arguments[argument] = flat.name
                parameters = {k:v for k,v in p.items() if k != "weight_layout"}
                calls.append(replace(call, parameters={**parameters, "type": 1}, arguments=tuple(arguments)))
            graph.calls = calls
        if settings.get("stage_inputs"):
            from akas.graph import Call
            calls = []
            for index, call in enumerate(graph.calls):
                p = call.parameters
                if (call.kind not in ("linear", "linear_add", "ffn")
                        or not p.get("half_accum") or p["r"] == 1):
                    calls.append(call)
                    continue
                if call.bindings or p.get("input_dtype", "float32") != "float32":
                    raise ValueError("input staging requires positional F32 projections")
                staged = graph.buffer(f"pipeline.input.{index}", (p["r"] * p["k"],), "float16")
                calls.append(Call("linear_half_input", dict(r=p["r"], k=p["k"]),
                                  (call.arguments[0], staged.name)))
                calls.append(replace(call, parameters={**p, "input_dtype": "float16"},
                                     arguments=(staged.name, *call.arguments[1:])))
            graph.calls = calls
        return graph, commit
    if set(settings) & {"split_ffn", "split_ffn_min_rows", "stage_weights", "stage_inputs"}:
        raise ValueError("prefill projection policies require the floating contract")
    if graph.contract != "block-fp8":
        return (graph, commit)
    if provider != "cuda":
        raise ValueError("native FP8 projection pipelines require CUDA")
    if settings.get("hopper") and target != "sm_90a":
        raise ValueError("Hopper projection policy requires an sm_90a target")
    partitions = settings.get("partitions", 1)
    if type(partitions) is not int or partitions < 1:
        raise ValueError("projection partitions must be positive")
    recorder = Graph(contract=graph.contract, phase=graph.phase)
    recorder.buffers = graph.buffers
    quantized, valid = ({}, set())
    calls = []
    for index, call in enumerate(graph.calls):
        if call.kind != "fp8_linear":
            calls.append(call)
            for name in (call.bindings.get("out"),):
                valid.discard(name)
            continue
        if call.arguments:
            names = dict(zip(("x", "w", "scales", "out"), call.arguments))
        else:
            names = dict(call.bindings)
        p = call.parameters
        count, depth, width = (p["r"], p["k"], p["o"])
        parts = min(partitions, depth // 128)
        if depth % (128 * parts):
            raise ValueError("split projection requires whole block-128 partitions")
        if "activation_scales" not in names:
            source = names["x"]
            if source not in quantized:
                storage = get("activations", "fp8").storage(
                    graph.buffers[source].shape, provider=provider
                )
                bits = graph.buffer(
                    f"pipeline.{source}.bits",
                    storage.values.shape,
                    storage.values.dtype,
                )
                scales = graph.buffer(
                    f"pipeline.{source}.scales",
                    storage.metadata[0].shape,
                    storage.metadata[0].dtype,
                )
                quantized[source] = (bits, scales)
            bits, scales = quantized[source]
            if source not in valid:
                recorder(
                    "quantize",
                    dict(r=count, k=depth),
                    x=graph.buffers[source],
                    out=bits,
                    scales=scales,
                )
                calls.extend(recorder.calls)
                recorder.calls = []
                valid.add(source)
            names.update(x=bits.name, activation_scales=scales.name)
        original = graph.buffers[names["out"]]
        partial = (
            original
            if parts == 1
            else graph.buffer(
                f"pipeline.partial.{index}", (count, parts, width), "float32"
            )
        )
        parameters = {
            **p,
            "partitions": parts,
            "columns": 64,
            "block_m": 16,
            "threads": 128,
            "stages": 1,
            "packed_copy": False,
            **settings,
        }
        parameters["partitions"] = parts
        recorder(
            "projection_partial",
            parameters,
            **{
                name: graph.buffers[value]
                for name, value in {**names, "out": partial.name}.items()
            },
            label=call.label if parts == 1 else None,
        )
        if parts > 1:
            recorder(
                "projection_merge",
                dict(r=count, o=width, partitions=parts),
                partial=partial,
                out=original,
                label=call.label,
            )
        calls.extend(recorder.calls)
        recorder.calls = []
        valid.discard(original.name)
    graph.calls = calls
    return (graph, commit)


def project(graph, x, weight, *, width, name, role="scratch"):
    """Declare dense projection scratch from the selected operation geometry."""
    from akas.ops.linear.definition import schedule
    from akas.quantization.weights import Weight

    if isinstance(weight, Weight):
        rows, depth = x.shape
        columns, expected_depth = weight.shape
        if depth != expected_depth:
            raise ValueError("projection input depth differs from its encoded weight")
        out = graph.buffer(name, (rows, columns), "float32", role=role)
        graph(
            "quantized_projection",
            dict(
                r=rows,
                k=depth,
                o=columns,
                input_dtype=x.dtype,
                weight_parts=weight.parts,
            ),
            x,
            *weight.buffers,
            out,
        )
        return out

    rows, depth = x.shape
    columns, expected_depth = weight.shape
    if depth != expected_depth:
        raise ValueError("projection input depth differs from its weight")
    out = graph.buffer(name, (rows, columns), "float32", role=role)
    parameters = schedule(rows, depth, columns, width)
    partial = (
        out.view((rows, 1, columns))
        if parameters["parts"] == 1
        else graph.buffer(
            name + ".parts", (rows, parameters["parts"], columns), "float32"
        )
    )
    graph("split_linear", parameters, x, weight, partial)
    if parameters["parts"] > 1:
        graph("split_merge", parameters, partial, out)
    return out


def linear(emit, x, weight, out, *, rows, scales=None, activation_scales=None):
    columns, depth = weight.shape
    parameters = dict(r=rows, k=depth, o=columns)
    bindings = dict(x=x, w=weight, out=out)
    if scales is not None:
        bindings["scales"] = scales
        if activation_scales is not None:
            bindings["activation_scales"] = activation_scales
    elif activation_scales is not None:
        raise ValueError("activation scales require FP8 weight scales")
    emit("bf16_linear" if scales is None else "fp8_linear", parameters, **bindings)


def packed_linear(emit, inputs, weight, out, *, rows, shape, quantization, tile):
    columns, depth = shape
    parameters = dict(r=rows, k=depth, o=columns, type=quantization, **tile)
    arguments = inputs("linear", parameters)
    emit("linear", parameters, *arguments, weight, out)


def split_linear(emit, device, x, weight, out, *, rows, parameters, scratch, views):
    """Emit split/merge calls, retaining a borrowed view for singleton splits."""
    if parameters["parts"] == 1:
        scratch = out.view((rows, 1, weight.shape[0]))
        views.append(scratch)
    emit("split_linear", parameters, x, weight, scratch)
    if parameters["parts"] > 1:
        emit("split_merge", parameters, scratch, out)


def fuse_residual_projection(graph):
    """Join packed decode projection and its private residual addition."""
    uses = Counter(
        (
            name
            for call in graph.calls
            for name in (*call.arguments, *call.bindings.values())
        )
    )
    aliases = {value.alias_of for value in graph.buffers.values() if value.alias_of}
    result, index = ([], 0)
    while index < len(graph.calls):
        pair = graph.calls[index : index + 2]
        if len(pair) == 2:
            projection, addition = pair
            p = projection.parameters
            if (
                projection.kind == "linear"
                and addition.kind == "add"
                and p.get("ggml")
                and (p["r"] == 1)
                and (len(projection.arguments) == len(addition.arguments) == 3)
                and (addition.parameters == dict(r=1, c=p["o"]))
            ):
                temporary = projection.arguments[-1]
                if (
                    temporary in addition.arguments[:2]
                    and uses[temporary] == 2
                    and (temporary not in aliases)
                    and (graph.buffers[temporary].role == "scratch")
                ):
                    residual = (
                        addition.arguments[1]
                        if addition.arguments[0] == temporary
                        else addition.arguments[0]
                    )
                    result.append(
                        replace(
                            projection,
                            kind="linear_add",
                            arguments=(
                                *projection.arguments[:2],
                                residual,
                                addition.arguments[-1],
                            ),
                            label=addition.label,
                        )
                    )
                    index += 2
                    continue
        result.append(graph.calls[index])
        index += 1
    graph.calls = result
    used = {
        name for call in result for name in (*call.arguments, *call.bindings.values())
    }
    graph.buffers = {
        name: spec
        for name, spec in graph.buffers.items()
        if spec.role != "scratch" or name in used or name in aliases
    }
    return graph


def fuse_projections(plan):
    """Fuse private paired projections into the established FFN operation.

    Decode with one row keeps separate projections: the established FP16 GEMV
    route is faster there. Partitioned or externally consumed intermediates
    keep their original calls. The runtime recomposes this same graph.
    """
    from collections import Counter
    from dataclasses import replace

    uses = Counter(
        (
            name
            for call in plan.calls
            for name in (*call.arguments, *call.bindings.values())
        )
    )
    aliases = {value.alias_of for value in plan.buffers.values() if value.alias_of}
    result, index = ([], 0)
    while index < len(plan.calls):
        calls = plan.calls[index : index + 3]
        if len(calls) == 3:
            gate, up, activation = calls
            p = gate.parameters
            if (
                (gate.kind, up.kind, activation.kind) == ("linear", "linear", "swiglu")
                and p == up.parameters
                and p.get("optimized")
                and (p["r"] > 1 or (p.get("ggml") and p.get("type") not in (0, 1)))
                and (not p.get("matrix_output"))
                and (p.get("output_offset", 0) == 0)
                and (p.get("output_stride", p["o"]) == p["o"])
                and (
                    len(gate.arguments)
                    == len(up.arguments)
                    == len(activation.arguments)
                    == 3
                )
                and (gate.arguments[0] == up.arguments[0])
                and (activation.arguments[:2] == (gate.arguments[2], up.arguments[2]))
                and (activation.parameters == dict(r=p["r"], c=p["o"]))
                and (uses[gate.arguments[2]] == uses[up.arguments[2]] == 2)
                and all(
                    (
                        plan.buffers[name].role == "scratch" and name not in aliases
                        for name in (gate.arguments[2], up.arguments[2])
                    )
                )
            ):
                result.append(
                    replace(
                        gate,
                        kind="ffn",
                        arguments=(
                            gate.arguments[0],
                            gate.arguments[1],
                            up.arguments[1],
                            activation.arguments[2],
                        ),
                        label=activation.label,
                    )
                )
                index += 3
                continue
        result.append(plan.calls[index])
        index += 1
    plan.calls = result
    used = {
        name for call in result for name in (*call.arguments, *call.bindings.values())
    }
    plan.buffers = {
        name: spec
        for name, spec in plan.buffers.items()
        if spec.role != "scratch" or name in used or name in aliases
    }
    return plan


def schedule(rows, k, o, width):
    m = 64 if rows >= 128 else 32 if rows >= 32 and o > width else 16
    return dict(
        r=rows, k=k, o=o, m=m, n=64, bk=128, parts=8 if o == width and k > width else 1
    )


REWRITES = (fuse_projections, fuse_residual_projection)
ROW_INDEPENDENT = ("linear",)
OUTPUT_ARGUMENTS = {"reference": {"linear": (-1,), "ffn": (-1,), "linear_add": (-1,)}}
OUTPUT_ARGUMENTS["reference"]["linear_half_input"] = OutputABI((-1,), ("out",))
from .validation import half_input_validation

VALIDATIONS["linear_half_input"] = half_input_validation
for _contract in ("reference", "packed", "packed-baseline", "bf16", "block-fp8"):
    OUTPUT_ARGUMENTS.setdefault(_contract, {}).update(
        {
            "linear": OutputABI((-1,), ("out",)),
            "gemv": OutputABI((-1,), ("out",)),
            "ffn": OutputABI((-1,), ("out",)),
            "linear_add": OutputABI((-1,), ("out",)),
            "bf16_linear": OutputABI((-1,), ("out",)),
            "fp8_linear": OutputABI((-1,), ("out",)),
            "split_linear": OutputABI((-1,), ("out",)),
            "split_merge": OutputABI((-1,), ("out",)),
            "projection_partial": OutputABI((-1,), ("out",)),
            "projection_merge": OutputABI((-1,), ("out",)),
        }
    )
VALIDATIONS.update(
    {name: pipeline_validation for name in ("projection_partial", "projection_merge")}
)
IMPLEMENTATIONS = (
    Implementation(
        "linear_half_input", "akas.ops.linear.webgpu", "reference", provider="webgpu",
        factory="half_input", name="linear.webgpu.half_input",
    ),
    Implementation(
        "fp8_linear",
        "akas.ops.linear.cuda",
        "block-fp8",
        phase="decode",
        priority=20,
        defaults=(("block_m", 16), ("columns", 64), ("threads", 128)),
        accepts=lambda p: p.get("r", 1) > 8,
        factory="fp8_inline",
        name="linear.cuda.block_fp8",
    ),
    Implementation(
        "linear",
        "akas.ops.linear.cuda",
        "packed",
        priority=20,
        space=(
            ("threads", (64, 128, 256)),
            ("unroll", (1, 2, 4, 8)),
            ("f16_values", (4, 8, 16)),
        ),
        factory="linear",
        name="linear.cuda.packed",
    ),
    Implementation(
        "linear_add",
        "akas.ops.linear.cuda",
        "packed",
        priority=20,
        factory="linear_add",
        name="linear.cuda.packed",
    ),
    Implementation(
        "ffn",
        "akas.ops.linear.cuda",
        "packed",
        priority=20,
        factory="ffn",
        name="linear.cuda.packed",
    ),
    Implementation(
        "linear",
        "akas.ops.linear.kernel",
        "packed-baseline",
        factory="packed_linear",
        name="linear.cuda.packed_baseline",
    ),
    Implementation(
        "linear",
        "akas.ops.linear.kernel",
        "packed",
        priority=0,
        factory="packed_linear",
        name="linear.cuda.packed_baseline",
    ),
    Implementation(
        "linear",
        "akas.ops.linear.kernel",
        "packed",
        provider="webgpu",
        priority=0,
        factory="packed_linear",
        name="linear.cuda.packed_baseline",
    ),
    Implementation(
        "linear",
        "akas.ops.linear.webgpu",
        "packed",
        provider="webgpu",
        priority=20,
        factory="kernel_linear",
        name="linear.webgpu.packed",
    ),
    Implementation(
        "ffn",
        "akas.ops.linear.webgpu",
        "packed",
        provider="webgpu",
        priority=20,
        factory="ffn",
        name="linear.webgpu.packed",
    ),
    Implementation(
        "linear_q16",
        "akas.ops.linear.webgpu",
        "packed",
        provider="webgpu",
        priority=20,
        factory="linear_q16",
        name="linear.webgpu.packed",
    ),
    Implementation(
        "ffn_q16",
        "akas.ops.linear.webgpu",
        "packed",
        provider="webgpu",
        priority=20,
        factory="ffn_q16",
        name="linear.webgpu.packed",
    ),
    Implementation(
        "linear_add",
        "akas.ops.linear.webgpu",
        "packed",
        provider="webgpu",
        priority=20,
        factory="linear_add",
        name="linear.webgpu.packed",
    ),
    Implementation(
        "linear_q8",
        "akas.ops.linear.webgpu",
        "packed",
        provider="webgpu",
        priority=20,
        accepts=lambda p: p.get("r", 1) == 1 and p["k"] % 128 == 0,
        factory="linear_q8",
        name="linear.webgpu.packed",
    ),
    Implementation(
        "linear",
        "akas.ops.linear.cuda",
        "bf16",
        space=(
            ("m", (16, 32, 64)),
            ("n", (64, 128)),
            ("bk", (128, 64, 256)),
            ("stages", (2, 1, 3)),
        ),
        semantic_keys=("r", "k", "o"),
        factory="linear_bf16",
        name="linear.cuda.bf16_batch",
    ),
    Implementation(
        "gemv",
        "akas.ops.linear.cuda",
        "bf16",
        space=(("columns", (16, 8, 32)),),
        semantic_keys=("r", "k", "o"),
        factory="gemv_linear_bf16",
        name="linear.cuda.bf16_batch",
    ),
    Implementation(
        "bf16_linear",
        "akas.ops.linear.cuda",
        "block-fp8",
        factory="gemv_native_bf16",
        name="linear.cuda.decode",
    ),
    Implementation(
        "fp8_linear",
        "akas.ops.linear.cuda",
        "block-fp8",
        phase="decode",
        factory="native_fp8",
        name="linear.cuda.decode",
    ),
    Implementation(
        "bf16_linear",
        "akas.ops.linear.cuda",
        "block-fp8",
        phase="prefill",
        priority=30,
        accepts=lambda p: not p.get("decode_head"),
        factory="native_bf16",
        name="linear.cuda.block_fp8",
    ),
    Implementation(
        "fp8_linear",
        "akas.ops.linear.cuda",
        "block-fp8",
        phase="prefill",
        defaults=(
            ("columns", 64),
            ("block_m", 16),
            ("threads", 128),
            ("packed_copy", True),
            ("stages", 2),
        ),
        space=(
            ("block_m", (16, 32, 64)),
            ("columns", (64, 32, 128)),
            ("threads", (128, 256)),
            ("stages", (2, 1, 3)),
            ("packed_copy", (True, False)),
        ),
        factory="fp8_prequantized",
        name="linear.cuda.block_fp8",
    ),
    Implementation(
        "split_linear",
        "akas.ops.linear.cuda",
        "bf16",
        space=(
            ("m", (16, 32, 64)),
            ("n", (64, 128)),
            ("bk", (128, 256)),
            ("threads", (128, 256)),
        ),
        semantic_keys=("r", "k", "o", "parts"),
        factory="partial",
        name="linear.cuda.split",
    ),
    Implementation(
        "split_merge",
        "akas.ops.linear.cuda",
        "bf16",
        factory="split",
        name="linear.cuda.split",
    ),
    Implementation(
        "linear",
        "akas.ops.linear.kernel",
        "reference",
        space=(("threads", (32, 64, 128, 256)),),
        accepts=lambda p: not p.get("ggml", False),
        factory="linear",
        name="linear.reference",
    ),
    Implementation(
        "linear",
        "akas.ops.linear.kernel",
        "reference",
        provider="webgpu",
        space=(("threads", (32, 64, 128, 256)),),
        accepts=lambda p: not p.get("ggml", False),
        factory="linear",
        name="linear.reference",
    ),
    Implementation(
        "linear",
        "akas.ops.linear.webgpu",
        "reference",
        provider="webgpu",
        priority=20,
        space=(("threads", (32, 64, 128, 256)),),
        accepts=lambda p: not p.get("ggml", False),
        factory="linear",
        name="linear.webgpu.reference",
    ),
    Implementation(
        "linear",
        "akas.ops.linear.cuda",
        "reference",
        priority=30,
        space=(
            ("minimum_mma_groups", (32, 0, 64, 128, 256)),
            ("maximum_gemv_rows", (31, 24, 8, 16, 25, 28, 32, 64)),
            ("block_m", (32, 16, 64)),
            ("block_n", (64, 32, 128)),
            ("block_k", (32, 64, 128)),
        ),
        accepts=lambda p: (
            not p.get("ggml", False)
            and p.get("optimized", False)
            and (p["k"] % 32 == 0)
        ),
        factory="floating_linear",
        name="linear.cuda.floating",
    ),
    Implementation(
        "ffn",
        "akas.ops.linear.cuda",
        "reference",
        priority=30,
        space=(
            ("minimum_mma_groups", (32, 0, 64, 128, 256)),
            ("maximum_gemv_rows", (31, 24, 8, 16, 25, 28, 32, 64)),
            ("block_m", (32, 16, 64)),
            ("block_n", (64, 32, 128)),
            ("block_k", (32, 64, 128)),
        ),
        accepts=lambda p: (
            not p.get("ggml", False)
            and p.get("optimized", False)
            and (p["k"] % 32 == 0)
        ),
        factory="floating_ffn",
        name="linear.cuda.floating",
    ),
    Implementation(
        "linear",
        "akas.ops.linear.webgpu",
        "reference",
        provider="webgpu",
        priority=30,
        space=(
            ("minimum_mma_groups", (0, 64, 128, 256, 512, 1024)),
            ("schedule", ("register", "outer")),
            ("outer_shared_dtype", ("float16", "float32")),
        ),
        accepts=lambda p: (
            not p.get("ggml", False)
            and p.get("optimized", False)
            and (p["k"] % 32 == 0)
        ),
        factory="floating_linear",
        name="linear.webgpu.floating",
    ),
    Implementation(
        "ffn",
        "akas.ops.linear.webgpu",
        "reference",
        provider="webgpu",
        priority=30,
        space=(
            ("minimum_mma_groups", (0, 64, 128, 256, 512, 1024)),
            ("schedule", ("register", "outer")),
            ("outer_shared_dtype", ("float16", "float32")),
        ),
        accepts=lambda p: (
            not p.get("ggml", False)
            and p.get("optimized", False)
            and (p["k"] % 32 == 0)
        ),
        factory="floating_ffn",
        name="linear.webgpu.floating",
    ),
    Implementation(
        "linear",
        "akas.ops.linear.cuda",
        "reference",
        priority=40,
        space=(
            ("threads", (64, 128, 256)),
            ("unroll", (1, 2, 4, 8)),
            ("f16_values", (4, 8, 16)),
        ),
        semantic_keys=("r", "k", "o", "type", "ggml"),
        accepts=lambda p: p.get("ggml", False) and p["r"] == 1,
        factory="linear",
        name="linear.cuda.packed",
    ),
    Implementation(
        "ffn",
        "akas.ops.linear.cuda",
        "reference",
        priority=40,
        space=(
            ("threads", (64, 128, 256)),
            ("unroll", (1, 2, 4, 8)),
            ("f16_values", (4, 8, 16)),
        ),
        semantic_keys=("r", "k", "o", "type", "ggml"),
        accepts=lambda p: p.get("ggml", False) and p["r"] == 1,
        factory="ffn",
        name="linear.cuda.packed",
    ),
    Implementation(
        "linear_add",
        "akas.ops.linear.cuda",
        "reference",
        priority=40,
        space=(
            ("threads", (64, 128, 256)),
            ("unroll", (1, 2, 4, 8)),
            ("f16_values", (4, 8, 16)),
        ),
        semantic_keys=("r", "k", "o", "type", "ggml"),
        accepts=lambda p: p.get("ggml", False) and p["r"] == 1,
        factory="linear_add",
        name="linear.cuda.packed",
    ),
    Implementation(
        "linear",
        "akas.ops.linear.cuda",
        "reference",
        priority=40,
        defaults=(("block_m", 32), ("block_n", 64), ("block_k", 32), ("stages", 2)),
        space=(
            ("block_m", (32, 16, 64, 128)),
            ("block_n", (64, 32, 128, 256)),
            ("block_k", (32, 64, 128)),
            ("stages", (2, 1, 3)),
            ("threads", (128, 256)),
        ),
        semantic_keys=("r", "k", "o", "type", "ggml"),
        variant="prefill",
        accepts=lambda p: p.get("ggml", False) and p["r"] > 1,
        factory="linear",
        name="linear.cuda.packed:prefill",
    ),
    Implementation(
        "ffn",
        "akas.ops.linear.cuda",
        "reference",
        priority=40,
        defaults=(("block_m", 32), ("block_n", 64), ("block_k", 32), ("stages", 2)),
        space=(
            ("block_m", (32, 16, 64, 128)),
            ("block_n", (64, 32, 128, 256)),
            ("block_k", (32, 64, 128)),
            ("stages", (2, 1, 3)),
            ("threads", (128, 256)),
        ),
        semantic_keys=("r", "k", "o", "type", "ggml"),
        variant="prefill",
        accepts=lambda p: p.get("ggml", False) and p["r"] > 1,
        factory="ffn",
        name="linear.cuda.packed:prefill",
    ),
    Implementation(
        "linear_add",
        "akas.ops.linear.cuda",
        "reference",
        priority=40,
        defaults=(("block_m", 32), ("block_n", 64), ("block_k", 32), ("stages", 2)),
        space=(
            ("block_m", (32, 16, 64, 128)),
            ("block_n", (64, 32, 128, 256)),
            ("block_k", (32, 64, 128)),
            ("stages", (2, 1, 3)),
            ("threads", (128, 256)),
        ),
        semantic_keys=("r", "k", "o", "type", "ggml"),
        variant="prefill",
        accepts=lambda p: p.get("ggml", False) and p["r"] > 1,
        factory="linear_add",
        name="linear.cuda.packed:prefill",
    ),
    Implementation(
        "linear",
        "akas.ops.linear.webgpu",
        "reference",
        provider="webgpu",
        priority=40,
        accepts=lambda p: (
            p.get("ggml", False) and (p["r"] != 1 or p["type"] not in (2, 14))
        ),
        factory="kernel_linear",
        name="linear.webgpu.packed",
    ),
    Implementation(
        "ffn",
        "akas.ops.linear.webgpu",
        "reference",
        provider="webgpu",
        priority=40,
        accepts=lambda p: (
            p.get("ggml", False) and (p["r"] != 1 or p["type"] not in (2, 14))
        ),
        factory="ffn",
        name="linear.webgpu.packed",
    ),
    Implementation(
        "linear_add",
        "akas.ops.linear.webgpu",
        "reference",
        provider="webgpu",
        priority=40,
        accepts=lambda p: (
            p.get("ggml", False) and (p["r"] != 1 or p["type"] not in (2, 14))
        ),
        factory="linear_add",
        name="linear.webgpu.packed",
    ),
    Implementation(
        "linear",
        "akas.ops.linear.webgpu",
        "reference",
        provider="webgpu",
        priority=40,
        space=(
            ("sg", (False, True)),
            ("gemv_threads", (128, 64, 256)),
            ("gemv_lanes", (16, 32)),
            ("gemv_unroll", (1, 2, 4, 8)),
            ("gemv_chains", (1, 2, 4)),
        ),
        semantic_keys=("r", "k", "o", "type", "ggml"),
        variant="quantized-decode",
        accepts=lambda p: (
            p.get("ggml", False) and p["r"] == 1 and (p["type"] in (2, 14))
        ),
        factory="kernel_linear",
        name="linear.webgpu.packed:quantized-decode",
    ),
    Implementation(
        "ffn",
        "akas.ops.linear.webgpu",
        "reference",
        provider="webgpu",
        priority=40,
        space=(
            ("sg", (False, True)),
            ("gemv_threads", (128, 64, 256)),
            ("gemv_lanes", (16, 32)),
            ("gemv_unroll", (1, 2, 4, 8)),
            ("gemv_chains", (1, 2, 4)),
        ),
        semantic_keys=("r", "k", "o", "type", "ggml"),
        variant="quantized-decode",
        accepts=lambda p: (
            p.get("ggml", False) and p["r"] == 1 and (p["type"] in (2, 14))
        ),
        factory="ffn",
        name="linear.webgpu.packed:quantized-decode",
    ),
    Implementation(
        "linear_add",
        "akas.ops.linear.webgpu",
        "reference",
        provider="webgpu",
        priority=40,
        space=(
            ("sg", (False, True)),
            ("gemv_threads", (128, 64, 256)),
            ("gemv_lanes", (16, 32)),
            ("gemv_unroll", (1, 2, 4, 8)),
            ("gemv_chains", (1, 2, 4)),
        ),
        semantic_keys=("r", "k", "o", "type", "ggml"),
        variant="quantized-decode",
        accepts=lambda p: (
            p.get("ggml", False) and p["r"] == 1 and (p["type"] in (2, 14))
        ),
        factory="linear_add",
        name="linear.webgpu.packed:quantized-decode",
    ),
    Implementation(
        "projection_partial",
        "akas.ops.linear.cuda",
        "block-fp8",
        space=(
            ("columns", (64, 32, 128)),
            ("block_m", (16, 32, 64)),
            ("threads", (128, 256)),
            ("stages", (1, 2, 3)),
        ),
        semantic_keys=("r", "k", "o", "partitions"),
        factory="fp8_prequantized",
        name="linear.pipeline",
    ),
    Implementation(
        "projection_merge",
        "akas.ops.linear.cuda",
        "block-fp8",
        factory="fp8",
        name="linear.pipeline",
    ),
    Implementation(
        "projection_partial",
        "akas.ops.linear.cuda",
        "block-fp8",
        priority=20,
        targets=("sm_90a",),
        space=(
            ("columns", (64, 128)),
            ("block_m", (16, 32, 64)),
            ("threads", (128, 256)),
            ("stages", (1, 2, 3)),
        ),
        semantic_keys=("r", "k", "o", "partitions"),
        accepts=lambda p: p.get("hopper", False),
        factory="hopper_gemm_partial",
        name="linear.hardware.hopper.projection",
    ),
    Implementation(
        "bf16_linear",
        "akas.ops.linear.cuda",
        "block-fp8",
        phase="decode",
        priority=20,
        variant="large-batch",
        accepts=lambda p: p.get("r", 1) > 8 and p["k"] % 64 == 0,
        factory="native_bf16",
        name="linear.cuda.block_fp8:large-batch",
    ),
    Implementation(
        "bf16_linear",
        "akas.ops.linear.cuda",
        "block-fp8",
        phase="prefill",
        priority=20,
        required=("decode_head",),
        variant="large-head",
        accepts=lambda p: (
            p.get("decode_head") and p.get("r", 1) > 8 and (p["k"] % 64 == 0)
        ),
        factory="native_bf16",
        name="linear.cuda.block_fp8:large-head",
    ),
)


IMPLEMENTATIONS += tuple(
    Implementation(
        "quantized_linear",
        f"akas.ops.linear.{provider}",
        "quantized",
        factory="quantized",
        provider=provider,
        required=("r", "k", "o"),
    )
    for provider in ("cuda", "webgpu")
)
OUTPUT_ARGUMENTS["quantized"] = {"quantized_linear": OutputABI((-1,), ("out",))}
from akas.ops.linear.validation import quantized_validation

VALIDATIONS["quantized_linear"] = quantized_validation
OUTPUT_ARGUMENTS["reference"]["quantized_linear"] = OutputABI((-1,), ("out",))
IMPLEMENTATIONS += (
    Implementation(
        "quantized_linear",
        "akas.ops.linear.cuda",
        "reference",
        factory="quantized",
        name="linear.cuda.channel_fp8",
        required=("r", "k", "o"),
        accepts=lambda p: (
            p.get("weight_format") == "fp8"
            and p.get("weight_options", {}).get("scale_layout") == "channel"
        ),
    ),
)

IMPLEMENTATIONS += tuple(
    Implementation(
        "quantized_projection",
        f"akas.ops.linear.{provider}",
        "*",
        factory="projection",
        provider=provider,
        accepts=lambda p, provider=provider, tiled=tiled: (
            (p["r"] > 1) == tiled
            and (p.get("input_dtype", "bfloat16") != "bfloat16" or provider == "cuda")
        ),
        defaults=tuple(
            zip(
                ("block_m", "block_n", "block_k"),
                (32, 64, 32) if provider == "cuda" else (16, 32, 32),
            )
        )
        if tiled
        else (),
        space=(("block_m", (16, 32)), ("block_n", (32, 64)), ("block_k", (32, 64)))
        if tiled and provider == "cuda"
        else (("block_k", (32, 64)),)
        if tiled
        else (),
        semantic_keys=("r", "k", "o", "input_dtype", "weight_parts"),
        variant="prefill" if tiled else "decode",
    )
    for provider in ("cuda", "webgpu")
    for tiled in (False, True)
)
# Packed prefill previously exposed only the register schedule to automatic
# search. Keep that incumbent and independently qualify staged weight reuse.
IMPLEMENTATIONS += tuple(
    Implementation(
        kind, "akas.ops.linear.webgpu", "reference", provider="webgpu",
        priority=39, factory="outer_prefill",
        defaults=(("outer_rows", 32),),
        space=(("outer_rows", (32, 64, 16)),),
        semantic_keys=("r", "k", "o", "type", "ggml"),
        accepts=lambda p: p.get("ggml", False) and p["r"] > 1 and p["type"] in (1, 2, 14),
        variant="outer-prefill", name="linear.webgpu.packed:outer-prefill",
    )
    for kind in ("linear", "ffn")
)

OUTPUT_ARGUMENTS["*"] = {"quantized_projection": OutputABI((-1,), ("out",))}
from .validation import projection_validation

VALIDATIONS["quantized_projection"] = projection_validation

IMPLEMENTATIONS += tuple(
    Implementation(
        kind, "akas.ops.linear.webgpu", "reference", provider="webgpu",
        priority=43, factory="integer_prefill",
        defaults=(("integer_layout", 0), ("integer_k", 32)),
        space=(("integer_layout", tuple(range(5))), ("integer_k", (32, 64, 128))),
        semantic_keys=("r", "k", "o", "type", "ggml", "q16"),
        accepts=lambda p: p.get("ggml", False) and p.get("q16", False)
        and p["r"] > 1 and p["type"] == 2,
        name="linear.webgpu.two_component",
    )
    for kind in ("linear", "ffn", "linear_add")
)

IMPLEMENTATIONS += tuple(
    Implementation(
        kind, "akas.ops.linear.webgpu", "reference", provider="webgpu",
        priority=43, factory="half_prefill",
        defaults=(("half_layout", 0), ("half_pad", 1), ("half_native", True), ("half_k", 16)),
        space=(("half_layout", (0, 1, 2, 3, 4, 5)), ("half_pad", (1, 0)),
               ("half_native", (True, False)), ("half_k", (16, 32, 64))),
        semantic_keys=("r", "k", "o", "type", "ggml", "half_accum"),
        accepts=lambda p: p.get("ggml", False) and p.get("half_accum", False)
        and p["r"] > 1 and p["type"] in (1, 2, 14),
        name="linear.webgpu.short_half",
    )
    for kind in ("linear", "ffn", "linear_add")
)

IMPLEMENTATIONS += tuple(
    Implementation(
        kind, "akas.ops.linear.webgpu", "reference", provider="webgpu",
        priority=38, factory="blocked_prefill",
        defaults=(("outer_layout", 0),),
        space=(("outer_layout", tuple(range(9))),),
        semantic_keys=("r", "k", "o", "type", "ggml"),
        accepts=lambda p: p.get("ggml", False) and p["r"] > 1 and p["type"] in (1, 2, 14),
        variant="blocked-prefill", name="linear.webgpu.packed:blocked-prefill",
    )
    for kind in ("linear", "ffn", "linear_add")
)

IMPLEMENTATIONS += tuple(
    Implementation(
        kind, "akas.ops.linear.webgpu", "reference", provider="webgpu",
        priority=39, factory="vector_decode",
        defaults=(("decode_layout", 0),), space=(("decode_layout", tuple(range(8))),),
        semantic_keys=("r", "k", "o", "type", "ggml"),
        accepts=lambda p: p.get("ggml", False) and p["r"] == 1 and p["type"] in (2, 14),
        variant="vector-decode", name="linear.webgpu.packed:vector-decode",
    )
    for kind in ("linear", "ffn", "linear_add")
)

IMPLEMENTATIONS += tuple(
    Implementation(
        kind, "akas.ops.linear.webgpu", "reference", provider="webgpu",
        priority=37, factory="branched_prefill",
        defaults=(("outer_layout", 0),), space=(("outer_layout", tuple(range(9))),),
        semantic_keys=("r", "k", "o", "type", "ggml"),
        accepts=lambda p: p.get("ggml", False) and p["r"] > 1 and p["type"] in (1, 2, 14),
        variant="branch-prefill", name="linear.webgpu.packed:branch-prefill",
    )
    for kind in ("linear", "ffn", "linear_add")
)

IMPLEMENTATIONS += tuple(
    Implementation(
        kind, "akas.ops.linear.webgpu", "reference", provider="webgpu",
        priority=37, factory="archived_decode",
        semantic_keys=("r", "k", "o", "type", "ggml"),
        accepts=lambda p: p.get("ggml", False) and p["r"] == 1 and p["type"] in (2, 14),
        variant="archived-decode", name="linear.webgpu.packed:archived-decode",
    )
    for kind in ("linear", "ffn", "linear_add")
)

IMPLEMENTATIONS += tuple(
    Implementation(
        kind, "akas.ops.linear.webgpu", "reference", provider="webgpu",
        priority=36, factory="padded_prefill",
        defaults=(("shared_layout", 0),), space=(("shared_layout", (0, 1, 2, 3)),),
        semantic_keys=("r", "k", "o", "type", "ggml"),
        accepts=lambda p: p.get("ggml", False) and p["r"] >= 128 and p["type"] in (1, 2, 14),
        variant="padded-prefill", name="linear.webgpu.packed:padded-prefill",
    )
    for kind in ("linear", "ffn", "linear_add")
)

IMPLEMENTATIONS += (
    Implementation(
        "linear",
        "akas.ops.linear.cuda",
        "reference",
        factory="floating_warp",
        name="linear.cuda.floating_warp",
        priority=25,
        defaults=(("threads", 128),),
        space=(("threads", (128, 256)),),
        accepts=lambda p: (
            "type" not in p and p.get("weight_dtype", "float32") == "float32"
        ),
    ),
)


# Independent columns expose small F32 projections to more workgroups. CUDA
# retains its established warp specialization until this alternative is timed.
IMPLEMENTATIONS += tuple(
    Implementation(
        "linear",
        f"akas.ops.linear.{provider}",
        "reference",
        provider=provider,
        factory="column_projection",
        name=f"linear.{provider}.column_projection",
        priority=30 if provider == "webgpu" else -10,
        defaults=(("threads", 256),),
        space=(("threads", (32, 64, 128, 256)),),
        accepts=lambda p: (
            p["r"] == 1
            and p["k"] >= 128
            and p["o"] <= 256
            and not p.get("ggml", False)
            and "type" not in p
            and p.get("weight_dtype", "float32") == "float32"
        ),
    )
    for provider in ("cuda", "webgpu")
)

IMPLEMENTATIONS += (
    Implementation(
        "linear",
        "akas.ops.linear.cuda",
        "reference",
        factory="q8_linear",
        name="linear.cuda.q8",
        priority=45,
        accepts=lambda p: (
            p.get("vector_q8", False)
            and p.get("ggml", False)
            and p.get("type") == 8
            and p["r"] == 1
            and p["k"] % 256 == 0
        ),
        space=(("threads", (128, 256)), ("unroll", (4, 2, 8))),
    ),
    Implementation(
        "linear",
        "akas.ops.linear.cuda",
        "reference",
        factory="short_packed_linear",
        name="linear.cuda.short_packed",
        priority=46,
        targets=("sm_89",),
        accepts=lambda p: (
            p.get("short_packed", False)
            and p.get("ggml", False)
            and 1 < p["r"] <= 64
            and p["k"] % 512 == 0
            and p.get("type") in (8, 12, 14, 30)
        ),
    ),
)
