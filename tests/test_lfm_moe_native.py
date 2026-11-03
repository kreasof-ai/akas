"""Original FP8 safetensors use shared model/runtime and numerical operations."""

import json
import os
from pathlib import Path
import struct

import numpy as np
import pytest

from akas.models.lfm2 import LFM2MoE
from akas.models.lfm2.config import LFM2MoEConfig
from akas.models.lfm2.weights import LFM2MoECheckpoint


def fixture(directory):
    from akas.models.lfm2.weights import huggingface_names, tensor_shapes
    from akas.quantization._shared.e4m3 import encode_storage
    from tensor.runtime.dtypes import encode_bfloat16

    directory.mkdir(parents=True)
    config = dict(model_type="lfm2_moe", hidden_size=256, intermediate_size=256,
                  num_attention_heads=4, num_key_value_heads=1, num_hidden_layers=2,
                  layer_types=["conv", "full_attention"], vocab_size=64, conv_L_cache=3,
                  max_position_embeddings=64, norm_eps=1e-5, num_experts=4,
                  num_experts_per_tok=2, moe_intermediate_size=256, num_dense_layers=1,
                  use_expert_bias=True, norm_topk_prob=True, routed_scaling_factor=1.0,
                  quantization_config={"quant_method": "compressed-tensors", "format": "float-quantized",
                      "config_groups": {"FP8_BLOCK": {
                          "weights": dict(num_bits=8, type="float", strategy="block", block_structure=[128, 128],
                                          dynamic=False, symmetric=True),
                          "input_activations": dict(num_bits=8, type="float", strategy="group", group_size=128,
                                                    dynamic=True, symmetric=True)}}})
    c = LFM2MoEConfig.from_huggingface(config)
    shapes = tensor_shapes(c)
    names = huggingface_names(c)
    sources = {name: (names[name],) for name in shapes if name in names}
    sources["blk.1.ffn_gate_inp.weight"] = ("model.layers.1.feed_forward.gate.weight",)
    sources["blk.1.exp_probs_b.bias"] = ("model.layers.1.feed_forward.expert_bias",)
    for part, original in (("gate", "w1"), ("up", "w3"), ("down", "w2")):
        sources[f"blk.1.ffn_{part}_exps.weight"] = tuple(
            f"model.layers.1.feed_forward.experts.{e}.{original}.weight" for e in range(4))
    rng, payload, header = np.random.default_rng(93), bytearray(), {}

    def append(name, values, dtype):
        raw = encode_bfloat16(values).tobytes() if dtype == "BF16" else values.tobytes()
        start = len(payload)
        payload.extend(raw)
        header[name] = dict(shape=list(values.shape), dtype=dtype, data_offsets=[start, len(payload)])

    for name, shape in shapes.items():
        group = sources[name]
        local = shape[1:] if len(group) > 1 else shape
        if name.endswith("conv.weight"):
            local = (c.width, 1, c.conv)
        quantized = name.endswith("_exps.weight") or (len(local) == 2 and name not in
                    ("token_embd.weight", "blk.1.ffn_gate_inp.weight") and "shortconv." not in name)
        for source in group:
            values = (rng.uniform(.8, 1.2, local) if len(local) == 1 and not name.endswith("bias")
                      else rng.normal(0, .03, local)).astype("float32")
            if quantized:
                append(source, encode_storage(values / .01), "F8_E4M3")
                append(source[:-6] + "weight_scale", np.full(((local[0] + 127) // 128, local[1] // 128), .01,
                                                             dtype="float32"), "BF16")
            else:
                append(source, values, "F32" if name.endswith("bias") else "BF16")
    encoded = json.dumps(header).encode()
    (directory / "config.json").write_text(json.dumps(config))
    (directory / "model.safetensors").write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)
    return directory


def test_native_moe_maps_original_scale_buffers_and_shared_graph(tmp_path):
    from akas.optimization.context import discover

    path = fixture(tmp_path / "weights")
    loaded = LFM2MoECheckpoint(path, tokenizer=False)
    assert discover(path).endswith(":LFM2MoE")
    assert loaded.config.experts == 4 and loaded.config.top_k == 2
    graph = LFM2MoE.compose(loaded, rows=3, chunk=4, max_requests=4, max_context=64).graph
    assert graph.buffers["blk.1.ffn_gate_exps.weight"].shape == (4, 256, 256)
    assert graph.buffers["blk.1.ffn_gate_exps.weight.scale"].shape == (4, 2, 2)
    assert graph.buffers["blk.1.ffn_gate_exps.weight.scale"].dtype == "bfloat16"
    assert {"quantize", "expert_partial", "routed_combine", "fp8_linear"} <= {call.kind for call in graph.calls}
    assert any(call.kind == "expert_tile_map" for call in graph.calls)
    assert all(call.parameters.get("tile_map") for call in graph.calls if call.kind == "expert_partial")
    assert all(spec.dtype == "float16" for name, spec in graph.buffers.items() if name.endswith((".kc", ".vc")))
    with pytest.raises(ValueError, match="CUDA"):
        LFM2MoE.compose(loaded, rows=1, chunk=1, max_requests=1, max_context=64, provider="webgpu")


def test_compact_expert_policy_retiles_tables_and_consumers_together(tmp_path):
    from akas.compilation.policy import lower, candidates
    from akas.compilation.validate import kernel_bindings
    from akas.kernels.definition import define
    from tensor.compiler.frontend.signature import frontend_arguments

    checkpoint = LFM2MoECheckpoint(fixture(tmp_path / "weights"), tokenizer=False)
    graph = LFM2MoE.compose(checkpoint, rows=3, chunk=4, max_requests=4, max_context=64).graph
    assert {"moe": {"block_m": 64}} in candidates(graph, provider="cuda", target="sm_89")
    graph, _ = lower(graph, policy={"moe": {"block_m": 64}}, provider="cuda", target="sm_89")
    for call in graph.calls:
        if call.kind not in ("expert_partial", "expert_tiles"):
            continue
        assert call.parameters["block_m"] == 64
        primitive = define(dict(kind=call.kind, parameters=call.parameters, contract="reference"))
        kernel_bindings(graph, call.key, frontend_arguments(primitive, {}))


@pytest.mark.parametrize("visible", [None, "output", "alias", "extra_user"])
@pytest.mark.parametrize("minimum", [1, 13])
def test_fp8_expert_fusion_requires_private_intermediates(tmp_path, visible, minimum):
    from dataclasses import replace
    from akas.compilation.policy import lower, candidates
    from akas.compilation.compile import plan

    checkpoint = LFM2MoECheckpoint(fixture(tmp_path / "weights"), tokenizer=False)
    graph = LFM2MoE.compose(checkpoint, rows=3, chunk=4, max_requests=4, max_context=64).graph
    gate = next(call for call in graph.calls if call.kind == "expert_partial")
    name = gate.bindings["out"]
    if visible == "output":
        graph.buffers[name] = replace(graph.buffers[name], role="output")
    elif visible == "alias":
        graph.view("observed_gate", graph.buffers[name], graph.buffers[name].shape)
    elif visible == "extra_user":
        graph.calls.append(gate)
    if visible is None:
        assert {"moe": {"fused_swiglu": True}} in candidates(graph, provider="cuda", target="sm_89")
    graph, _ = lower(graph, policy={"moe": {"fused_swiglu": True, "block_m": 32,
                                           "fused_quantize_minimum_rows": minimum}})
    fused = [call for call in graph.calls if call.kind == ("fp8_expert_swiglu" if minimum == 1 else "expert_swiglu")]
    assert len(fused) == (1 if visible is None else 0)
    if fused:
        assert fused[0].parameters["block_m"] == 32
        assert graph.buffers[fused[0].bindings["out"]].dtype == ("uint8" if minimum == 1 else "float32")
        plan(graph)


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("fused,minimum", [(False, 1), (True, 1), (True, 100)])
@pytest.mark.parametrize("visible", [None, "feature", "alias", "extra_user"])
def test_private_expert_activation_precision_survives_fusion(tmp_path, dtype, fused, minimum, visible):
    from akas.compilation.policy import lower
    from akas.compilation.compile import plan

    checkpoint = LFM2MoECheckpoint(fixture(tmp_path / "weights"), tokenizer=False)
    graph = LFM2MoE.compose(checkpoint, rows=3, chunk=4, max_requests=4, max_context=64).graph
    original = next(call for call in graph.calls if call.kind == "swiglu_experts")
    name = original.arguments[-1]
    if visible == "feature":
        graph.request.setdefault("features", {})["activation"] = name
    elif visible == "alias":
        graph.view("borrowed_activation", graph.buffers[name], graph.buffers[name].shape)
    elif visible == "extra_user":
        graph.calls.append(original)
    settings = dict(activation_dtype=dtype, fused_swiglu=fused)
    if fused:
        settings["fused_quantize_minimum_rows"] = minimum
    graph, _ = lower(graph, policy={"moe": settings})
    if visible is not None:
        assert graph.buffers[name].dtype == "float32"
        assert not any(call.kind in ("expert_swiglu", "fp8_expert_swiglu") for call in graph.calls)
    elif fused:
        call = next(call for call in graph.calls if call.kind == ("fp8_expert_swiglu" if minimum == 1 else "expert_swiglu"))
        assert call.parameters["product_dtype"] == call.parameters["activation_dtype"] == dtype
        assert graph.buffers[call.bindings["out"]].dtype == ("uint8" if minimum == 1 else dtype)
    else:
        activation = next(call for call in graph.calls if call.kind == "swiglu_experts")
        assert activation.parameters["input_dtype"] == activation.parameters["output_dtype"] == dtype
        assert all(graph.buffers[name].dtype == dtype for name in activation.arguments)
    plan(graph, target="sm_89")


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("visible", [None, "feature", "extra_user"])
def test_routed_contribution_storage_changes_private_aliases_and_both_operations(tmp_path, dtype, visible):
    from akas.compilation.policy import lower
    from akas.compilation.compile import plan

    checkpoint = LFM2MoECheckpoint(fixture(tmp_path / "weights"), tokenizer=False)
    graph = LFM2MoE.compose(checkpoint, rows=3, chunk=4, max_requests=4, max_context=64).graph
    down = next(call for call in graph.calls if call.kind == "expert_partial" and call.parameters.get("routed_input"))
    name = down.bindings["out"]
    borrowed = graph.view("observed_contributions", graph.buffers[name], graph.buffers[name].shape)
    combine = next(call for call in graph.calls if call.kind == "routed_combine")
    if visible == "feature":
        graph.request.setdefault("features", {})["observed"] = borrowed.name
    elif visible == "extra_user":
        graph.calls.append(combine)
    graph, _ = lower(graph, policy={"moe": {"fused_swiglu": True, "intermediate_dtype": dtype}})
    expected = dtype if visible is None else "float32"
    assert graph.buffers[name].dtype == graph.buffers[borrowed.name].dtype == expected
    down = next(call for call in graph.calls if call.kind == "expert_partial" and call.parameters.get("routed_input"))
    combine = next(call for call in graph.calls if call.kind == "routed_combine")
    assert down.parameters.get("output_dtype", "float32") == expected
    assert combine.parameters.get("input_dtype", "float32") == expected
    plan(graph, target="sm_89")


def test_expert_policy_extends_producer_and_every_consumer_map_contract(tmp_path):
    from akas.compilation.policy import lower
    from akas.compilation.compile import plan

    checkpoint = LFM2MoECheckpoint(fixture(tmp_path / "weights"), tokenizer=False)
    graph = LFM2MoE.compose(checkpoint, rows=3, chunk=4, max_requests=4, max_context=64).graph
    graph, _ = lower(graph, policy={"moe": {"max_block_m": 128, "block_m": 32, "fused_swiglu": True}})
    producer = next(call for call in graph.calls if call.kind == "expert_tile_map")
    assert producer.parameters["max_block_m"] == 128
    assert all(graph.buffers[name].shape[0] == 6 for name in producer.arguments[1:])
    consumers = [call for call in graph.calls if call.kind in ("fp8_expert_swiglu", "expert_partial")]
    assert consumers and all(call.parameters["max_block_m"] == 128 for call in consumers)
    plan(graph, target="sm_89")


@pytest.mark.parametrize("mutation", [
    lambda d: d.update(num_experts_per_tok=5),
    lambda d: d.update(use_expert_bias=False),
    lambda d: d["quantization_config"]["config_groups"]["FP8_BLOCK"]["weights"].update(block_structure=[64, 128]),
])
def test_native_moe_rejects_unsupported_semantics_before_allocating(tmp_path, mutation):
    path = fixture(tmp_path / "weights")
    data = json.loads((path / "config.json").read_text())
    mutation(data)
    (path / "config.json").write_text(json.dumps(data))
    with pytest.raises(ValueError):
        LFM2MoECheckpoint(path, tokenizer=False)


@pytest.mark.skipif(os.getenv("AKAS_LFM_FP8_CUDA") != "1", reason="native FP8 CUDA qualification")
@pytest.mark.parametrize("kind,p", [
    ("fp8_linear", dict(r=3, k=256, o=128, input_dtype="float32")),
    ("expert_partial", dict(r=3, rows=3, k=256, o=128, experts=4, top=2, partitions=1, block_m=16)),
    ("expert_partial", dict(r=1, rows=1, k=256, o=128, experts=64, top=4, partitions=1, block_m=16, compact=True)),
    ("expert_partial", dict(r=33, rows=33, k=256, o=128, experts=4, top=2, partitions=1, block_m=16, compact=True)),
    ("expert_partial", dict(r=33, rows=33, k=256, o=128, experts=4, top=2, partitions=1, block_m=64, compact=True)),
    ("expert_partial", dict(r=33, rows=33, k=384, o=128, experts=4, top=2, partitions=1, block_m=32, compact=True, stages=2)),
    ("expert_partial", dict(r=33, rows=33, k=384, o=128, experts=4, top=2, partitions=1, block_m=32, compact=True, stages=3, routed_input=True)),
    ("expert_partial", dict(r=33, rows=33, k=384, o=128, experts=4, top=2, partitions=1, block_m=32, compact=True, activation_vector=16)),
    ("expert_partial", dict(r=33, rows=33, k=384, o=128, experts=4, top=2, partitions=1, block_m=32, compact=True, stages=2, activation_vector=16, routed_input=True)),
    ("expert_tiles", dict(r=33, experts=4, top=2, block_m=16)),
    ("expert_tiles", dict(r=1, experts=64, top=4, block_m=16)),
    ("fp8_expert_swiglu", dict(r=1, rows=1, k=256, o=128, experts=7, top=3, partitions=1, block_m=16, compact=True)),
    ("fp8_expert_swiglu", dict(r=33, rows=33, k=384, o=256, experts=7, top=3, partitions=1, block_m=32, compact=True, stages=2, activation_vector=16)),
    ("fp8_expert_swiglu", dict(r=33, rows=33, k=384, o=256, experts=7, top=3, partitions=1, block_m=16, compact=True, tile_map=True, activation_vector=16)),
    ("fp8_expert_swiglu", dict(r=33, rows=33, k=384, o=256, experts=7, top=3, partitions=1, block_m=32, compact=True, tile_map=True, activation_vector=16, pair_layout="separate")),
    ("fp8_expert_swiglu", dict(r=33, rows=33, k=384, o=256, experts=7, top=3, partitions=1, block_m=64, compact=True, tile_map=True, activation_vector=16)),
    ("expert_swiglu", dict(r=1, rows=1, k=256, o=256, experts=7, top=3, partitions=1, block_m=16, compact=True, tile_map=True, input_dtype="uint8", weight_format="fp8", columns=32, stages=2)),
    ("expert_swiglu", dict(r=33, rows=33, k=384, o=256, experts=7, top=3, partitions=1, block_m=32, compact=True, tile_map=True, input_dtype="uint8", weight_format="fp8", columns=64, stages=2)),
    ("fp8_expert_swiglu", dict(r=33, rows=33, k=384, o=256, experts=7, top=3, partitions=1, block_m=16, compact=True, tile_map=True, activation_vector=16, stages=1, mma_dtype="float16")),
    ("expert_swiglu", dict(r=33, rows=33, k=384, o=256, experts=7, top=3, partitions=1, block_m=32, compact=True, tile_map=True, input_dtype="uint8", weight_format="fp8", columns=32, stages=1, mma_dtype="float16")),
    ("expert_partial", dict(r=33, rows=33, k=384, o=256, experts=7, top=3, partitions=1, block_m=32, compact=True, tile_map=True, routed_input=True, columns=64, stages=1, mma_dtype="float16")),
    ("expert_partial", dict(r=33, rows=33, k=384, o=256, experts=7, top=3, partitions=1, block_m=32, compact=True, tile_map=True, routed_input=True, columns=64, stages=1, mma_dtype="float16", output_dtype="bfloat16")),
    ("routed_combine", dict(r=3, top=4, c=256, input_dtype="bfloat16")),
    ("expert_tile_map", dict(r=257, experts=7, top=3, max_block_m=128)),
    ("fp8_expert_swiglu", dict(r=257, rows=257, k=384, o=256, experts=7, top=3, partitions=1, block_m=128, max_block_m=128, compact=True, tile_map=True, activation_vector=16, threads=512, pair_layout="separate", product_dtype="bfloat16", activation_dtype="bfloat16")),
    ("expert_partial", dict(r=257, rows=257, k=384, o=256, experts=7, top=3, partitions=1, block_m=128, max_block_m=128, compact=True, tile_map=True, activation_vector=16, threads=512, output_dtype="bfloat16", routed_input=True)),
    ("fp8_expert_swiglu", dict(r=33, rows=33, k=384, o=256, experts=7, top=3, partitions=1, block_m=32, compact=True, tile_map=True, activation_vector=16, product_dtype="bfloat16", activation_dtype="bfloat16")),
    ("fp8_expert_swiglu", dict(r=33, rows=33, k=384, o=256, experts=7, top=3, partitions=1, block_m=32, compact=True, tile_map=True, activation_vector=16, product_dtype="float16", activation_dtype="float16", pair_layout="separate")),
    ("expert_swiglu", dict(r=33, rows=33, k=384, o=256, experts=7, top=3, partitions=1, block_m=32, compact=True, tile_map=True, input_dtype="uint8", weight_format="fp8", columns=64, activation_vector=16, product_dtype="bfloat16", activation_dtype="bfloat16", output_dtype="bfloat16")),
    ("expert_partial", dict(r=33, rows=33, k=384, o=128, experts=7, top=3, partitions=1, block_m=32, compact=True, tile_map=True, activation_vector=16, routed_input=True)),
    ("expert_tile_map", dict(r=33, experts=7, top=3)),
    ("quantize", dict(r=3, k=256, input_dtype="float32")),
    ("conv", dict(r=3, c=128, length=3, chunk=2048)),
    ("expert_swiglu", dict(r=1, rows=1, k=384, o=256, experts=7, top=3,
        partitions=1, block_m=16, compact=True, tile_map=True,
        input_dtype="uint8", weight_format="fp8", columns=16, threads=128,
        pair_layout="wide", stages=2, activation_vector=16)),
    ("expert_swiglu", dict(r=4, rows=4, k=384, o=256, experts=7, top=3,
        partitions=1, block_m=16, compact=True, tile_map=True,
        input_dtype="uint8", weight_format="fp8", columns=16, threads=64,
        pair_layout="wide", stages=2, activation_vector=16)),
])
def test_native_operations_match_independent_references(tmp_path, kind, p):
    import tensor
    from akas.compilation.selection import source
    from akas.kernels.definition import define
    from akas.kernels.registry import validation
    from tensor.compiler.frontend.signature import frontend_arguments

    primitive = define(dict(kind=kind, parameters=p, contract="reference"))
    case = validation(kind, p, frontend_arguments(primitive, {}))
    entry, artifact = tmp_path / "kernel.py", tmp_path / "kernel.tbin"
    entry.write_text(source(kind, p, contract="reference", target="sm_89").render())
    tensor.build(entry, artifact, target="sm_89")
    with tensor.Device() as device:
        executable = device.load(artifact)
        buffers = {}
        for item in executable.manifest["arguments"]:
            values = case.arguments[item["name"]]
            buffers[item["name"]] = (device.from_numpy(values, dtype="bfloat16") if item["dtype"] == "bfloat16"
                                      else device.from_numpy(values))
        executable.launch(*(buffers[item["name"]] for item in executable.manifest["arguments"]))
        for name, expected in case.expected.items():
            np.testing.assert_allclose(buffers[name].to_numpy(), expected, rtol=case.rtol, atol=case.atol)
        if kind in ("fp8_expert_swiglu", "expert_swiglu") and not p.get("packed_input"):
            # Arbitrary signed FP8 operands and independently varying block scales
            # must retain the existing GPU sequence's exact quantization decisions.
            rng = np.random.default_rng(903)

            def upload(name, array):
                dtype = buffers[name].dtype
                buffers[name].release()
                buffers[name] = device.from_numpy(array, dtype=dtype)

            for name in ("x", "w", "w2"):
                array = rng.integers(32, 81, case.arguments[name].shape, dtype="uint8")
                array |= rng.integers(0, 2, array.shape, dtype="uint8") * 128
                upload(name, array)
            from tensor.runtime.dtypes import decode_bfloat16, encode_bfloat16
            for name in ("activation_scales", "scales", "scales2"):
                array = rng.uniform(.0625, .25, case.arguments[name].shape).astype("float32")
                upload(name,
                    decode_bfloat16(encode_bfloat16(array)) if name != "activation_scales" else array)

            def operation(op, parameters, arguments):
                entry = tmp_path / (op + ".py")
                artifact = tmp_path / (op + ".tbin")
                entry.write_text(source(op, parameters, contract="reference", target="sm_89").render())
                tensor.build(entry, artifact, target="sm_89")
                compiled = device.load(artifact)
                compiled.launch(*(arguments[a["name"]] for a in compiled.manifest["arguments"]))
                return compiled

            shape = (p["r"], p["top"], p["o"])
            product_dtype, activation_dtype = p.get("product_dtype", "float32"), p.get("activation_dtype", "float32")
            gate, up = (device.zeros(shape, dtype=product_dtype) for _ in range(2))
            activated = device.zeros(shape, dtype=activation_dtype)
            projection = operation("expert_partial", {**p, "columns": 128, "output_dtype": product_dtype}, {**buffers, "out": gate})
            up_arguments = {**buffers, "w": buffers["w2"], "scales": buffers["scales2"], "out": up}
            projection.launch(*(up_arguments[a["name"]] for a in projection.manifest["arguments"]))
            activation = operation("swiglu_experts", dict(r=p["r"], top=p["top"], c=p["o"],
                expert_rows=True, input_dtype=product_dtype, output_dtype=activation_dtype),
                dict(g=gate, u=up, out=activated))
            quantizer = None
            if kind == "fp8_expert_swiglu":
                quantizer = operation("quantize", dict(r=p["r"], top=p["top"], k=p["o"], input_dtype=activation_dtype),
                    dict(x=activated, out=buffers["out"], scales=buffers["output_scales"]))
                expected = {name: buffers[name].to_numpy() for name in ("out", "output_scales")}
            else:
                expected = {"out": activated.to_numpy()}
            executable.launch(*(buffers[a["name"]] for a in executable.manifest["arguments"]))
            for name, array in expected.items():
                np.testing.assert_array_equal(buffers[name].to_numpy(), array)
            for resource in (projection, activation, quantizer, gate, up, activated):
                if resource is None:
                    continue
                resource.close() if hasattr(resource, "close") else resource.release()


def reference(checkpoint, **options):
    from support.native_fp8_reference import NativeFP8Reference

    return NativeFP8Reference(checkpoint, **options)


def test_native_reference_keeps_router_operands_in_f32_when_mma_inputs_are_half(tmp_path):
    from support.native_fp8_reference import NativeFP8Reference

    checkpoint = LFM2MoECheckpoint(fixture(tmp_path / "weights"), tokenizer=False)
    oracle = NativeFP8Reference(checkpoint, projection_half=True)
    values = np.random.default_rng(31).normal(size=(3, checkpoint.config.width)).astype("float32")
    for name in ("blk.0.shortconv.in_proj.weight", "blk.1.ffn_gate_inp.weight"):
        source = values.astype("float16") if checkpoint.conversions[name] == "float16" else values
        expected = (source.astype("float64") @ oracle.weights[name].astype("float64").T).astype("float32")
        np.testing.assert_array_equal(oracle.project(name, values), expected)
    name = "blk.1.ffn_gate_inp.weight"
    rounded = (values.astype("float16").astype("float64") @ oracle.weights[name].T).astype("float32")
    assert not np.array_equal(oracle.project(name, values), rounded)


@pytest.mark.skipif(os.getenv("AKAS_LFM_FP8_CUDA") != "1", reason="native FP8 model qualification")
@pytest.mark.parametrize("block_m,fused_swiglu,projection_pipeline,fusion_minimum,contribution_dtype,activation_dtype", [
    (16, False, False, 1, "float32", "float32"), (32, False, False, 1, "float32", "float32"), (64, False, False, 1, "float32", "float32"),
    (32, True, False, 1, "float32", "float32"), (32, True, True, 1, "float32", "float32"), (32, True, True, 9, "float32", "float32"),
    (32, True, True, 9, "bfloat16", "float32"), (32, True, True, 9, "bfloat16", "bfloat16"),
    (128, True, True, 1, "bfloat16", "bfloat16")])
def test_native_model_reordered_ragged_requests_reuse_common_executor(tmp_path, block_m, fused_swiglu, projection_pipeline, fusion_minimum, contribution_dtype, activation_dtype, residual_dtype="float32", attention_half=False, packed_experts=False, expert_mma_dtype=None, attention_decode_half=False):
    import tensor
    from akas.compilation import compile_checkpoint
    from tensor.compiler.search import ScheduleProfile

    path = fixture(tmp_path / "weights")
    checkpoint = LFM2MoECheckpoint(path, tokenizer=False)
    bundle = tmp_path / "program"
    entries = [dict(operation="expert_partial", parameters=dict(compact=True),
                    schedule=dict(implementation="moe.cuda.fp8-bucketed:mapped",
                                  columns=128, threads=256, stages=2,
                                  activation_vector=16))] if block_m == 32 else []
    if projection_pipeline:
        entries.append(dict(operation="projection_partial", parameters={},
                            schedule=dict(implementation="linear.pipeline:reference", mma_dtype="float16")))
    if fusion_minimum > 1:
        entries.append(dict(operation="expert_swiglu", parameters=dict(weight_format="fp8"),
                            schedule=dict(implementation="moe.cuda.fp8-paired-swiglu:reference",
                                          columns=32, stages=2, activation_vector=16)))
    if block_m == 128:
        for operation, implementation in (("expert_partial", "moe.cuda.fp8-bucketed:mapped"),
                                          ("fp8_expert_swiglu", "moe.cuda.fp8-swiglu:mapped")):
            entries.append(dict(operation=operation, parameters=dict(compact=True), schedule=dict(
                implementation=implementation, block_m=128, threads=512,
                stages=1, activation_vector=16,
                **({"columns": 128} if operation == "expert_partial" else {"pair_layout": "separate"}))))
    if expert_mma_dtype is not None:
        for entry in entries:
            if entry["operation"] in ("expert_partial", "expert_swiglu", "fp8_expert_swiglu"):
                entry["schedule"]["mma_dtype"] = expert_mma_dtype
        for operation, implementation in (
                ("expert_partial", "moe.cuda.fp8-bucketed:mapped"),
                ("fp8_expert_swiglu", "moe.cuda.fp8-swiglu:mapped"),
                ("expert_swiglu", "moe.cuda.fp8-paired-swiglu:reference")):
            if not any(entry["operation"] == operation and entry["parameters"] == {}
                       for entry in entries):
                entries.append(dict(operation=operation, parameters={},
                                    schedule=dict(implementation=implementation, mma_dtype=expert_mma_dtype)))
    if attention_decode_half:
        entries.append(dict(operation="attention_partial",parameters=dict(decode_precision="float16"),
                            schedule=dict(implementation="attention.cuda.mma_decode",key_rows=64,
                                          copy_complete_tiles=True,stages=2)))
    schedules = ScheduleProfile(dict(
        schema="tensor.schedule-profile.v1", provider="cuda", target="sm_89",
        entries=entries,
    ))
    policy = {"moe": {"block_m": block_m, "fused_swiglu": fused_swiglu}}
    if attention_decode_half:
        policy["attention"] = dict(decode_precision="float16",splits=16,minimum_split_groups=256)
    if packed_experts:
        policy["moe"].update(packed=True,packed_minimum_rows=1)
    if block_m == 128:
        policy["moe"]["max_block_m"] = 128
    if contribution_dtype != "float32":
        policy["moe"]["intermediate_dtype"] = contribution_dtype
    if activation_dtype != "float32":
        policy["moe"]["activation_dtype"] = activation_dtype
    if fusion_minimum > 1:
        policy["moe"]["fused_quantize_minimum_rows"] = fusion_minimum
    if projection_pipeline:
        policy["linear"] = dict(partitions=8, minimum_rows=1 if fusion_minimum > 1 else 2, minimum_split_groups=64,
                                floating_matmul=True)
    if residual_dtype != "float32":
        policy = {"residual": dict(intermediate_dtype=residual_dtype), **policy,
                  "normalization": dict(fused_residual=True)}
    compile_checkpoint(LFM2MoE, path, bundle, checkpoint_options={"tokenizer": False},
                       geometries=[(1, 1), (2, 1), (2, 4)], max_requests=4, max_context=64,
                       target="sm_89", search=False,
                       policy=policy,
                       schedules=schedules)
    with tensor.Device() as device, LFM2MoE(path, bundle, device, checkpoint_options={"tokenizer": False}) as model:
        slots = [model.admit_at(3), model.admit_at(1)]
        sequences = [[1, 3, 5], [2, 4]]
        expected = [reference(checkpoint, contribution_dtype=contribution_dtype, activation_dtype=activation_dtype, residual_dtype=residual_dtype, attention_half=attention_half, attention_decode_half=attention_decode_half),
                    reference(checkpoint, contribution_dtype=contribution_dtype, activation_dtype=activation_dtype, residual_dtype=residual_dtype, attention_half=attention_half, attention_decode_half=attention_decode_half)]
        predictions = model.prefill(slots, sequences, chunk=4)
        actual = model.last_frame.feature("logits").to_numpy().reshape(2, 64)
        for row in range(2):
            target = expected[row].forward(sequences[row])
            np.testing.assert_allclose(actual[row], target, rtol=.05, atol=.005)
            assert predictions[row] == target.argmax()
        model.decode(slots[::-1], predictions[::-1])
        actual = model.last_frame.feature("logits").to_numpy().reshape(2, 64)
        for row, index in enumerate((1, 0)):
            target = expected[index].forward([predictions[index]])
            np.testing.assert_allclose(actual[row], target, rtol=.05, atol=.005)
        model.release(3)
        assert model.admit_at(3) == 3
        model.prefill([3], [[7, 9, 11]], chunk=4)
        reused = reference(checkpoint, contribution_dtype=contribution_dtype, activation_dtype=activation_dtype, residual_dtype=residual_dtype, attention_half=attention_half, attention_decode_half=attention_decode_half)
        target = reused.forward([7, 9, 11])
        np.testing.assert_allclose(model.last_frame.feature("logits").to_numpy().reshape(2, 64)[0],
                                   target, rtol=.05, atol=.005)
        model.decode([3], [int(target.argmax())])
        target = reused.forward([int(target.argmax())])
        np.testing.assert_allclose(model.last_frame.feature("logits").to_numpy().reshape(64),
                                   target, rtol=.05, atol=.005)


def test_compact_tile_capacity_bounds_routing_without_expert_count_padding():
    from akas.ops.moe.definition import tile_capacity

    rng = np.random.default_rng(29)
    assert tile_capacity(1, 4, 64, 16) == 4
    for _ in range(200):
        rows, experts, block_m = (int(rng.integers(1, 65)) for _ in range(3))
        top = int(rng.integers(1, experts + 1))
        counts = np.zeros(experts, dtype="int64")
        for _ in range(rows):
            counts[rng.choice(experts, top, replace=False)] += 1
        capacity = tile_capacity(rows, top, experts, block_m)
        assert int(((counts + block_m - 1) // block_m).sum()) <= capacity <= rows * top


def test_fused_fp8_reference_rejects_encoding_outside_f32_rounding_interval():
    from akas.kernels.definition import define
    from akas.kernels.registry import validation
    from tensor.compiler.frontend.signature import frontend_arguments
    from akas.quantization._shared.e4m3 import decode_storage

    p = dict(r=1, rows=1, k=128, o=128, experts=3, top=2,
             block_m=16, partitions=1, compact=True)
    primitive = define(dict(kind="fp8_expert_swiglu", parameters=p, contract="reference"))
    case = validation("fp8_expert_swiglu", p, frontend_arguments(primitive, {}))
    expected = case.expected["out"]
    compare = case.comparators["out"]
    compare(expected.copy(), expected)
    corrupted = expected.copy()
    index = np.abs(decode_storage(expected)).argmax()
    corrupted.reshape(-1)[index] ^= 128
    with pytest.raises(AssertionError):
        compare(corrupted, expected)
