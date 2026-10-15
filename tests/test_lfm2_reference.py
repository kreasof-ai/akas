"""Loader parity, independent numerical qualification, masks, reuse and targets."""

import json
import os
from pathlib import Path
import numpy as np
import pytest
from akas.models.lfm2.weights import ReferenceCheckpoint
from akas.models.lfm2.model import LFM2Reference
from akas.models.lfm2._composition import compose
from akas.compilation.compile import plan
from support.diagnostic import fixture
from support.checkpoint import export


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    directory = tmp_path_factory.mktemp("lfm-reference")
    gguf = fixture(directory / "model.gguf")
    return gguf, export(gguf, directory / "hf")


def test_formats_share_inventory_values_and_forward_plan(checkpoint):
    a, b = [ReferenceCheckpoint(p, tokenizer=False) for p in checkpoint]
    assert a.config == b.config
    assert a.names.keys() == b.names.keys()
    for name in a.names:
        np.testing.assert_array_equal(a.read(name), b.read(name))
    assert compose(a.config).metadata() == compose(b.config).metadata()


@pytest.mark.parametrize("dtype", ["F16", "BF16"])
def test_safetensors_native_float_formats(checkpoint, tmp_path, dtype):
    source = ReferenceCheckpoint(checkpoint[0], tokenizer=False)
    destination = ReferenceCheckpoint(
        export(checkpoint[0], tmp_path, dtype=dtype), tokenizer=False
    )
    for name in source.names:
        np.testing.assert_allclose(
            destination.read(name), source.read(name), rtol=0.004, atol=1e-5
        )


@pytest.mark.parametrize("provider", ["cuda", "webgpu"])
@pytest.mark.parametrize("batch", [1, 4, 16, 64, 256])
@pytest.mark.parametrize("kv_dtype", ["float16", "float32", "fp8"])
def test_all_reference_paths_select_without_model_compiler_branches(
    checkpoint, provider, batch, kv_dtype
):
    c = ReferenceCheckpoint(checkpoint[0], tokenizer=False).config
    forward = compose(
        c, batch=batch, capacity=384, provider=provider, kv_dtype=kv_dtype
    )
    bundle = plan(forward, provider=provider)
    assert len(bundle.kernels) == len(forward.requirements())
    assert all(
        kernel.source.module == "akas.kernels.definition" for kernel in bundle.kernels
    )
    cache = forward.buffers["blk.1.kc"]
    assert cache.shape == (
        batch,
        1,
        384,
        16 if provider == "webgpu" and kv_dtype == "fp8" else 64,
    )
    assert forward.buffers["blk.0.conv_state"].shape == (batch, 2, 256)


def test_configuration_rejects_unsupported_features(checkpoint, tmp_path):
    path = Path(checkpoint[1]) / "config.json"
    data = json.loads(path.read_text())
    from akas.models.lfm2.config import Config

    for field, value in (
        ("conv_bias", True),
        ("rope_scaling", {"type": "linear"}),
        ("num_key_value_heads", 3),
        ("layer_types", ["mamba", "attention"]),
    ):
        with pytest.raises(ValueError):
            Config.from_huggingface({**data, field: value})


@pytest.mark.parametrize("provider", ["cuda", "webgpu"])
@pytest.mark.parametrize("batch", [1, 4, 16, 64, 256])
@pytest.mark.parametrize("kv_dtype", ["float16", "fp8"])
def test_reference_executes_short_and_long_sequences(
    checkpoint, tmp_path, provider, batch, kv_dtype
):
    if os.environ.get("AKAS_LFM_REFERENCE_" + provider.upper()) != "1":
        pytest.skip("enable provider reference qualification")
    import tensor
    from akas.compilation import compile_checkpoint
    from support.oracle import Oracle

    path = os.environ.get("AKAS_LFM_REFERENCE_BUILD")
    bundle = Path(path or tmp_path) / f"{provider}-{kv_dtype}-b{batch}"
    capacity = 256
    compile_checkpoint(
        LFM2Reference,
        checkpoint[0],
        bundle,
        profile="reference",
        checkpoint_options={"tokenizer": False},
        batch=batch,
        capacity=capacity,
        provider=provider,
        kv_dtype=kv_dtype,
        max_weight_bytes=262144 if batch == 4 else 64 * 1024 * 1024,
    )
    with tensor.Device(provider=provider) as device:
        for source in checkpoint:
            with LFM2Reference(
                source, bundle, device, tokenizer=False, graphs=provider == "cuda"
            ) as model:
                rng = np.random.default_rng(43)
                prompts = [
                    rng.integers(0, model.config.vocab, 3 + i % 4, dtype="int32")
                    for i in range(batch)
                ]
                logits = model.prefill(prompts)
                for i in range(min(batch, 4)):
                    oracle = Oracle(source, context=capacity, kv_dtype=kv_dtype)
                    np.testing.assert_allclose(
                        logits[i], oracle.forward(prompts[i]), rtol=0.001, atol=0.0002
                    )
                # Decode uses each shorter slot's preserved sample, without host round trips.
                expected_first = logits.argmax(-1)
                before = model.positions.copy()
                sampled = model.decode(1)
                for i in range(min(batch, 4)):
                    oracle = Oracle(source, context=capacity, kv_dtype=kv_dtype)
                    oracle.forward(prompts[i])
                    expected = oracle.forward([int(expected_first[i])])
                    np.testing.assert_allclose(
                        model.logits()[i], expected, rtol=0.001, atol=0.0002
                    )
                    assert sampled[i] == expected.argmax()
                np.testing.assert_array_equal(model.positions, before + 1)
                # Reuse the same compiled graph at a much longer live context.
                model.reset()
                prompt = rng.integers(0, model.config.vocab, 128, dtype="int32")
                model.prefill([prompt] * batch, read=False)
                oracle = Oracle(source, context=capacity, kv_dtype=kv_dtype)
                actual, expected = model.logits()[0], oracle.forward(prompt)
                if kv_dtype == "fp8":
                    # Reduction order can cross BF16/FP8 rounding boundaries.
                    # Use a vector error bound instead of relative errors on
                    # individual logits near zero. Codec bytes are tested separately.
                    relative = np.linalg.norm(actual - expected) / max(
                        np.linalg.norm(expected), 1e-12
                    )
                    assert relative < 0.002
                    assert np.max(np.abs(actual - expected)) < 0.003
                    assert actual.argmax() == expected.argmax()
                else:
                    np.testing.assert_allclose(
                        actual, expected, rtol=0.002, atol=0.0003
                    )
                # State and positions for inactive slots stay bit-identical.
                state = model.buffers["blk.0.conv_state"].to_numpy()
                cache = model.buffers["blk.1.kc"].to_numpy()
                mask = np.zeros(batch, dtype="int32")
                mask[0] = 1
                model.decode(1, active=mask, read=False)
                np.testing.assert_array_equal(
                    model.buffers["blk.0.conv_state"].to_numpy()[1:], state[1:]
                )
                np.testing.assert_array_equal(
                    model.buffers["blk.1.kc"].to_numpy()[1:], cache[1:]
                )
                # Partial reset releases a slot's history without touching its siblings.
                model.reset([0])
                assert model.positions[0] == 0
                np.testing.assert_array_equal(
                    model.buffers["blk.0.conv_state"].to_numpy()[1:], state[1:]
                )
                with pytest.raises(ValueError, match="context"):
                    model.decode(capacity + 1)


@pytest.mark.parametrize("kv_dtype", ["float16", "fp8"])
def test_reference_builds_webgpu_without_device(checkpoint, tmp_path, kv_dtype):
    from akas.compilation import compile_checkpoint
    from tensor.artifacts.format import read_artifact

    result = compile_checkpoint(
        LFM2Reference,
        checkpoint[1],
        tmp_path,
        profile="reference",
        checkpoint_options={"tokenizer": False},
        batch=4,
        capacity=32,
        provider="webgpu",
        kv_dtype=kv_dtype,
        search=False,
    )
    import json

    for phase in result["phases"]:
        forward = json.loads((tmp_path / phase["path"]).read_text())
        for record in forward["kernels"].values():
            meta, _ = read_artifact(tmp_path / record["path"])
            assert meta["provider"] == "webgpu"


def test_partitioned_weight_upload_reassembles_original_values(checkpoint):
    from akas.checkpoints.partition import partitions

    c = ReferenceCheckpoint(checkpoint[0], tokenizer=False)

    class Device:
        def from_numpy(self, x):
            return np.array(x, copy=True)

    values = c.upload(Device(), max_weight_bytes=262144)
    for name in c.names:
        expected = c.read(name)
        parts = partitions(name, expected.shape, max_bytes=262144)
        restored = np.concatenate([values[p.name].reshape(p.shape) for p in parts])
        np.testing.assert_array_equal(restored, expected)
        assert all(values[p.name].nbytes <= 262144 for p in parts)
    assert len(values) > len(c.names)


def test_sharded_safetensors_index_and_extent_validation(checkpoint, tmp_path):
    import struct
    from akas.checkpoints.safetensors import tensor_index

    source = export(checkpoint[0], tmp_path)
    path = source / "model.safetensors"
    raw = path.read_bytes()
    size = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8 : 8 + size])
    index = {}
    for group in (0, 1):
        filename = f"model-{group}.safetensors"
        payload = bytearray()
        selected = {}
        for i, (name, info) in enumerate(header.items()):
            if i % 2 != group:
                continue
            a, b = info["data_offsets"]
            start = len(payload)
            payload.extend(raw[8 + size + a : 8 + size + b])
            selected[name] = {**info, "data_offsets": [start, len(payload)]}
            index[name] = filename
        encoded = json.dumps(selected).encode()
        (source / filename).write_bytes(
            struct.pack("<Q", len(encoded)) + encoded + payload
        )
    path.unlink()
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": index})
    )
    checkpoint_hf = ReferenceCheckpoint(source, tokenizer=False)
    checkpoint_gguf = ReferenceCheckpoint(checkpoint[0], tokenizer=False)
    assert tensor_index(source) == index
    for name in checkpoint_hf.names:
        np.testing.assert_array_equal(
            checkpoint_hf.read(name), checkpoint_gguf.read(name)
        )
    index[next(iter(index))] = "../outside.safetensors"
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": index})
    )
    with pytest.raises(ValueError, match="local file"):
        ReferenceCheckpoint(source, tokenizer=False)


@pytest.mark.parametrize("provider", ["cuda", "webgpu"])
@pytest.mark.parametrize("heads", [3, 96])
def test_reference_fp8_bytes_and_scales_match_torch(provider, heads, tmp_path):
    if os.environ.get("AKAS_LFM_REFERENCE_" + provider.upper()) != "1":
        pytest.skip("enable provider codec qualification")
    import torch
    import tensor
    from akas.compilation.emit import build_artifact
    from support.migration import source as export_kernel
    from akas.quantization.kv import get

    rows, d, cap = 4, 64, 8
    rng = np.random.default_rng(7)
    k = rng.normal(0, 1, (rows, heads * d)).astype("float32")
    v = rng.normal(0, 1, (rows, heads * d)).astype("float32")
    if heads == 96:
        factors = np.exp2(np.arange(heads) % 81 - 40).astype("float32")
        k *= np.repeat(factors, d)
        v *= np.repeat(factors, d)
    k[2, 0] = -0.0
    k[0, :d] = 0
    k[0, 0] = 448
    k[0, 1:8] = np.arange(1, 15, 2, dtype="float32") / 1024
    k[1, d : 2 * d] = 0  # exercise the minimum scale for an all-zero head
    positions = np.array([1, 3, 5, 7], dtype="int32")
    enabled = np.array([1, 1, 1, 0], dtype="int32")
    p = dict(
        r=rows, h=heads, kh=heads, d=d, capacity=cap, kv_dtype="fp8", provider=provider
    )
    source = tmp_path / "codec.py"
    source.write_text(
        export_kernel("akas.ops.attention.reference", "make_kernel", "cache_write", p)
    )
    artifact = tmp_path / "codec.tbin"
    build_artifact(
        source,
        artifact,
        target="sm_89" if provider == "cuda" else "webgpu-portable-v1",
        provider=provider,
    )
    storage = get("fp8").storage((rows, heads, cap, d), provider=provider)
    with tensor.Device(provider=provider) as device:
        buffers = []
        try:
            keys, values = [device.from_numpy(x) for x in (k, v)]
            buffers.extend([keys, values])
            payloads = [
                device.from_numpy(
                    np.zeros(storage.values.shape, dtype=storage.values.dtype)
                )
                for _ in range(2)
            ]
            metadata = [
                device.from_numpy(np.zeros(storage.metadata[0].shape, dtype="float32"))
                for _ in range(2)
            ]
            buffers.extend([*payloads, *metadata])
            controls = [device.from_numpy(x) for x in (positions, enabled)]
            buffers.extend(controls)
            kernel = device.load(artifact)
            try:
                kernel.launch(keys, values, *payloads, *metadata, *controls)
                for inputs, payload, scales in zip((k, v), payloads, metadata):
                    raw = payload.to_numpy()
                    raw = (
                        raw.view("uint8").reshape(rows, heads, cap, d)
                        if provider == "webgpu"
                        else raw
                    )
                    actual_scales = scales.to_numpy()
                    x = torch.from_numpy(inputs.reshape(rows, heads, d))
                    expected_scale = x.abs().amax(-1).clamp_min(1e-12) / 448
                    expected_bits = (
                        (x / expected_scale[..., None])
                        .to(torch.float8_e4m3fn)
                        .view(torch.uint8)
                        .numpy()
                    )
                    for row in range(rows - 1):
                        np.testing.assert_array_equal(
                            raw[row, :, positions[row]], expected_bits[row]
                        )
                        np.testing.assert_allclose(
                            actual_scales[row, :, positions[row], 0],
                            expected_scale[row].numpy(),
                            rtol=1e-7,
                            atol=0,
                        )
                    np.testing.assert_array_equal(raw[3], 0)
                    np.testing.assert_array_equal(actual_scales[3], 0)
            finally:
                kernel.release()
        finally:
            for buffer in reversed(buffers):
                buffer.release()


@pytest.mark.parametrize("provider", ["cuda", "webgpu"])
def test_partial_reset_preserves_sibling_generation_controls(
    provider, checkpoint, tmp_path
):
    if os.environ.get("AKAS_LFM_REFERENCE_" + provider.upper()) != "1":
        pytest.skip("enable provider reference qualification")
    import tensor
    from akas.compilation import compile_checkpoint

    bundle = (
        Path(os.environ.get("AKAS_LFM_REFERENCE_BUILD", str(tmp_path)))
        / f"{provider}-fp8-b4"
    )
    compile_checkpoint(
        LFM2Reference,
        checkpoint[0],
        bundle,
        profile="reference",
        checkpoint_options={"tokenizer": False},
        batch=4,
        capacity=256,
        kv_dtype="fp8",
        provider=provider,
        max_weight_bytes=262144,
    )
    with (
        tensor.Device(provider=provider) as device,
        LFM2Reference(
            checkpoint[0], bundle, device, tokenizer=False, graphs=True
        ) as model,
    ):
        model.prefill([[1, 2], [3, 4, 5], [6], [7, 8]])
        controls = {
            name: model.buffers[name].to_numpy()
            for name in ("tokens", "sampled", "active", "positions")
        }
        model.reset([0])
        for name, previous in controls.items():
            now = model.buffers[name].to_numpy()
            assert now[0] == 0
            np.testing.assert_array_equal(now[1:], previous[1:])
        result = model.generate([[1], [2], [3], [4]], max_tokens=3, chat=False)
        assert len(result) == 4 and all(
            len(row["generated_tokens"]) == 3 for row in result
        )
        assert model.positions.tolist() == [3] * 4
