"""Lossless Q4 layout permutations retain storage and independently read rows."""

import numpy as np
import pytest


def test_planar_roundtrip_preserves_every_scale_bit_and_source_byte():
    from akas.quantization.weights.ggml.kernel import repack_host

    rng = np.random.default_rng(29)
    original = rng.integers(0, 256, (65536, 18), dtype="uint8")
    original[:, :2] = np.arange(65536, dtype="uint16").view("uint8").reshape(-1, 2)
    before = original.copy()
    planar = repack_host(original)
    assert planar.nbytes == original.nbytes
    np.testing.assert_array_equal(planar[: 65536 * 16].reshape(-1, 16), original[:, 2:])
    np.testing.assert_array_equal(planar[65536 * 16 :].reshape(-1, 2), original[:, :2])
    np.testing.assert_array_equal(
        repack_host(planar, source="planar", target="raw").reshape(original.shape),
        original,
    )
    np.testing.assert_array_equal(original, before)


@pytest.mark.parametrize("provider", ["cuda", "webgpu"])
@pytest.mark.parametrize("shape", [(3, 256), (2, 7, 256), (4, 512)])
def test_planar_layout_keeps_allocation_and_bounded_reference_rows(provider, shape):
    from akas.quantization.formats import get
    from akas.quantization.weights.ggml.validation import (
        reference,
        sample_weight,
        decode_rows,
    )

    fmt = get("weights", "ggml")
    raw = fmt.storage(shape, provider=provider, encoding=2)
    packed = fmt.storage(shape, provider=provider, encoding=2, layout="planar")
    assert raw == packed
    a = np.empty(raw.values.shape, dtype=raw.values.dtype)
    b = np.empty_like(a)
    sample_weight(a, 2, seed=33, decode=False)
    read = reference(dict(encoding=2, layout="planar"), shape, b, seed=33)
    rows = int(np.prod(shape[:-1]))
    depth = shape[-1]
    for start, stop in ((0, 1), (rows - 1, rows), (0, rows)):
        np.testing.assert_array_equal(
            read(start, stop), decode_rows(a, 2, depth, start, stop)
        )
    assert fmt.supports("read", shape, provider=provider, encoding=2, layout="planar")
    assert fmt.supports(
        "read_tile", shape, provider=provider, encoding=2, layout="planar"
    )
    assert fmt.supports(
        "dot_float", shape, provider=provider, encoding=2, layout="planar"
    )
    assert fmt.supports(
        "dot_half", shape, provider=provider, encoding=2, layout="planar"
    ) == (provider == "cuda")


@pytest.mark.parametrize(
    "options", [dict(encoding=14, layout="planar"), dict(encoding=2, layout="unknown")]
)
def test_planar_rejects_ambiguous_or_unsupported_encoding(options):
    from akas.quantization.formats import get

    with pytest.raises(ValueError):
        get("weights", "ggml").storage((2, 256), provider="cuda", **options)


def test_planar_rejects_unsupported_dot_geometry_and_partial_blocks():
    from akas.quantization.formats import get
    from akas.quantization.weights.ggml.kernel import repack_host

    with pytest.raises(ValueError, match="divisible by 256"):
        get("weights", "ggml").storage(
            (2, 128), provider="cuda", encoding=2, layout="planar"
        )
    for data in (np.empty(0, dtype="uint8"), np.zeros(17, dtype="uint8")):
        with pytest.raises(ValueError, match="whole 18-byte blocks"):
            repack_host(data)


def test_lfm_planar_option_changes_only_encoded_expert_storage():
    from types import SimpleNamespace
    from akas.models.lfm2.config import LFM2MoEConfig
    from akas.models.lfm2.weights import LFM2MoECheckpoint, tensor_shapes
    from akas.models.lfm2.model import LFM2MoE
    from akas.ops.moe.definition import lower

    config = LFM2MoEConfig(
        width=256,
        ff=512,
        heads=4,
        kv_heads=1,
        layers=("conv", "attention", "conv"),
        vocab=512,
        conv=3,
        epsilon=1e-5,
        theta=1000000.0,
        max_context=512,
        experts=4,
        top_k=2,
        expert_width=256,
        dense_layers=2,
    )
    shapes = tensor_shapes(config)
    checkpoint = object.__new__(LFM2MoECheckpoint)
    checkpoint.config = config

    def encoding(name, shape):
        return (
            0
            if len(shape) == 1 or name.endswith(("conv.weight", "ffn_gate_inp.weight"))
            else 2
        )

    checkpoint.gguf = SimpleNamespace(
        tensors={
            name: SimpleNamespace(shape=shape, type=encoding(name, shape))
            for name, shape in shapes.items()
        }
    )
    graphs = {}
    for layout in ("raw", "planar"):
        composition = LFM2MoE.compose(
            checkpoint,
            rows=1,
            chunk=1,
            max_requests=1,
            max_context=512,
            provider="cuda",
            expert_weight_layout=layout,
        )
        graph, _ = lower(
            composition.graph,
            None,
            dict(fused_swiglu=True),
            provider="cuda",
            target="sm_89",
        )
        graphs[layout] = graph
    assert graphs["raw"].buffers == graphs["planar"].buffers
    assert graphs["raw"].request == graphs["planar"].request
    assert [c.kind for c in graphs["raw"].calls] == [
        c.kind for c in graphs["planar"].calls
    ]
    assert any(c.kind == "expert_swiglu" for c in graphs["planar"].calls)
    for raw, planar in zip(graphs["raw"].calls, graphs["planar"].calls):
        if raw.kind in ("quantized_experts", "expert_swiglu"):
            for field in ("weight_parts", "up_weight_parts"):
                if field in planar.parameters:
                    assert raw.parameters[field][0]["options"] == dict(encoding=2)
                    assert planar.parameters[field][0]["options"] == dict(
                        encoding=2, layout="planar"
                    )
        else:
            assert raw == planar
    assert (
        checkpoint.program_upload_options(graphs["raw"].buffers, model_options={}) == {}
    )
    assert checkpoint.program_upload_options(
        graphs["planar"].buffers, model_options=dict(expert_weight_layout="planar")
    ) == dict(expert_weight_layout="planar")


def test_packed_upload_reorders_only_selected_weights_and_keeps_source_bytes():
    from types import SimpleNamespace
    from akas.checkpoints.upload import upload_packed
    from akas.quantization.weights.ggml.kernel import repack_host

    raw = np.arange(144, dtype="uint8")
    native = np.arange(4, dtype="float32").view("uint8")
    before = raw.copy()
    arrays = {"expert": raw, "router": native}
    checkpoint = SimpleNamespace(
        tensors=dict(
            expert=SimpleNamespace(shape=(1, 256), type=2),
            router=SimpleNamespace(shape=(4,), type=0),
        ),
        packed=lambda n: arrays[n],
        discard=lambda n: None,
    )

    class Device:
        info = dict(provider="cuda")

        def from_numpy(self, array):
            return array.copy()

    out = upload_packed(
        Device(), checkpoint, layouts={"expert": "planar", "router": "raw"}
    )
    np.testing.assert_array_equal(out["expert"], repack_host(raw))
    np.testing.assert_array_equal(out["router"], native.view("float32"))
    np.testing.assert_array_equal(raw, before)


@pytest.mark.parametrize(
    "provider,rows,routed,rounded,kind",
    [
        ("cuda", 1, False, True, "quantized_experts"),
        ("cuda", 1, True, True, "quantized_experts"),
        ("cuda", 1, False, False, "quantized_experts"),
        ("cuda", 65, False, True, "quantized_experts"),
        ("cuda", 65, True, True, "quantized_experts"),
        ("cuda", 1, False, True, "expert_swiglu"),
        ("webgpu", 1, False, False, "quantized_experts"),
        ("webgpu", 1, True, False, "quantized_experts"),
        ("webgpu", 65, False, False, "quantized_experts"),
        ("webgpu", 65, True, False, "quantized_experts"),
    ],
)
def test_gpu_planar_experts_preserve_raw_outputs_and_inactive_contract(
    tmp_path, provider, rows, routed, rounded, kind
):
    import os

    if os.environ.get("AKAS_LFM_MOE_" + provider.upper()) != "1":
        pytest.skip("opt-in codec/operation GPU qualification")
    import tensor
    from tensor.artifacts.format import read_artifact
    from akas.compilation.emit import build_artifact
    from akas.compilation.selection import source
    from akas.kernels.registry import validation
    from akas.memory.transfer import write
    from akas.quantization.weights.ggml.kernel import repack_host

    base = dict(
        r=rows,
        top=4,
        experts=6,
        k=256,
        o=37,
        input_dtype="float32",
        round_inputs=rounded,
        routed_input=routed,
        tile_map=rows > 1,
    )
    artifacts = {}
    parameters = {}
    target = "sm_89" if provider == "cuda" else "webgpu-portable-v1"
    for layout in ("raw", "planar"):
        p = {
            **base,
            "weight_parts": (
                dict(rows=222, format="ggml", options=dict(encoding=2, layout=layout)),
            ),
        }
        if kind == "expert_swiglu":
            p["up_weight_parts"] = p["weight_parts"]
        parameters[layout] = p
        entry = tmp_path / (layout + ".py")
        artifact = entry.with_suffix(".tbin")
        entry.write_text(
            source(
                kind, p, contract="reference", provider=provider, target=target
            ).render()
        )
        build_artifact(entry, artifact, provider=provider, target=target)
        artifacts[layout] = artifact
    signature, _ = read_artifact(artifacts["raw"])
    case = validation(kind, parameters["raw"], signature["arguments"])
    results = {}
    with tensor.Device(provider=provider) as device:
        for layout, artifact in artifacts.items():
            arguments = {n: a.copy() for n, a in case.arguments.items()}
            if layout == "planar":
                for name in ("w", "w2"):
                    if name in arguments:
                        raw = arguments[name]
                        arguments[name] = (
                            repack_host(raw).view(raw.dtype).reshape(raw.shape)
                        )
            buffers = {n: device.from_numpy(a) for n, a in arguments.items()}
            try:
                with device.load(artifact) as module:
                    args = tuple(
                        buffers[a["name"]] for a in module.manifest["arguments"]
                    )
                    module.launch(*args)
                    actual = buffers["out"].to_numpy()
                    np.testing.assert_allclose(
                        actual, case.expected["out"], rtol=0.002, atol=0.0002
                    )
                    results[layout] = actual.copy()
                    for name, original in arguments.items():
                        if name != "out":
                            np.testing.assert_array_equal(
                                buffers[name].to_numpy(), original
                            )
                    write(device, buffers["counts"], np.zeros_like(arguments["counts"]))
                    module.launch(*args)
                    np.testing.assert_array_equal(
                        buffers["out"].to_numpy(),
                        np.zeros_like(actual) if rows == 1 else actual,
                    )
            finally:
                for buffer in buffers.values():
                    buffer.release()
    np.testing.assert_array_equal(
        results["raw"].view("uint32"), results["planar"].view("uint32")
    )
