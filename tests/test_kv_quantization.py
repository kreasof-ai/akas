"""Independent storage/codec extension and CUDA quantization qualification."""

import os
import sys
from types import SimpleNamespace
import numpy as np
import pytest

from akas.memory.state import BufferSpec
from akas.memory.layouts import hybrid_state_specs
from akas.quantization.kv import get, is_quantized
from akas.quantization.kv.codec import Format, Storage
from akas.ops.registry import select


@pytest.mark.parametrize("dim", [64, 128, 256, 512])
def test_fp8_storage_and_scale_layout_are_model_independent(dim):
    shape = (3, 5, 17, dim)
    storage = get("fp8").storage(shape)
    assert storage.values == BufferSpec(shape, "uint8")
    assert storage.metadata == (BufferSpec((3, 5, 17, (dim + 127) // 128), "float32"),)
    config = SimpleNamespace(layers=("attention",), kv_heads=5, head_dim=dim)
    assert hybrid_state_specs(config, 3, 17, kv_dtype="fp8")[0] == list(
        get("fp8").pair_layout(shape)
    )
    flat = hybrid_state_specs(config, 3, 17, kv_dtype="fp8", flat=True)[0]
    assert flat[0].shape == (3 * 5 * 17 * dim,)
    assert flat[2] == storage.metadata[0]


def test_another_codec_needs_no_central_dispatch_or_memory_change(monkeypatch):
    def layout(shape):
        return Storage(
            BufferSpec(shape[:-1] + ((shape[-1] + 1) // 2,), "uint8"),
            (BufferSpec(shape[:-1] + (2,), "float32"),),
        )

    codec = Format(
        "test_int4", layout, (("cuda", "akas.quantization.kv.fp8"),)
    )
    monkeypatch.setitem(
        sys.modules, "akas.quantization.kv.test_int4", SimpleNamespace(FORMAT=codec)
    )
    assert get("test_int4") is codec
    config = SimpleNamespace(layers=("attention",), kv_heads=3, head_dim=256)
    specs = hybrid_state_specs(config, 2, 11, kv_dtype="test_int4")[0]
    assert specs[0].shape == (2, 3, 11, 128)
    parameters = dict(r=2, capacity=11, kv_dtype="test_int4")
    assert (
        select(
            "attention_partial",
            parameters,
            contract="block-fp8",
            phase="decode",
            target="sm_89",
        ).module
        == "attention.cuda.quantized"
    )
    assert (
        select("kv_encode", parameters, contract="kv-pair", target="sm_89").module
        == "akas.quantization.kv.fp8"
    )
    # This test supplies storage/selection only; it does not claim an INT4 numerical implementation.


@pytest.mark.parametrize("name", ["missing", "../../fp8", "format", None])
def test_unknown_codec_rejected(name):
    with pytest.raises(ValueError):
        get(name)


def test_unquantized_storage_stays_separate_and_missing_provider_fails():
    assert not is_quantized("bfloat16")
    assert not is_quantized("float16")
    assert is_quantized("fp8")
    with pytest.raises(ValueError, match="does not support metal"):
        get("fp8").backend("metal")


GPU = pytest.mark.skipif(
    os.environ.get("TENSOR_QWEN_CUDA") != "1",
    reason="standalone CUDA KV codec qualification",
)


@GPU
@pytest.mark.parametrize("dim", [128, 256, 512])
@pytest.mark.parametrize("heads", [1, 3])
def test_standalone_codec_roundtrip_matches_torch_and_preserves_inactive_rows(
    tmp_path, dim, heads
):
    import tensor
    import torch
    from akas.compilation.emit import export_kernel

    torch.set_num_threads(1)
    slots, chunk, capacity = 2, 2, 16
    rows = slots * chunk
    rng = np.random.default_rng(912)
    k = rng.normal(size=(rows, heads * dim)).astype("float32")
    v = rng.normal(size=k.shape).astype("float32")
    positions = np.array([0, 7, 3, 11], dtype="int32")
    active = np.array([1, 0, 1, 1], dtype="int32")
    parameters = dict(
        slots=slots,
        chunk=chunk,
        kv_heads=heads,
        d=dim,
        capacity=capacity,
        kv_dtype="fp8",
    )
    artifacts = {}
    for kind in ("kv_encode", "kv_decode"):
        path = tmp_path / (kind + ".py")
        path.write_text(
            export_kernel(
                "akas.quantization.kv.fp8", "make_kernel", kind, parameters
            )
        )
        artifacts[kind] = path.with_suffix(".tbin")
        tensor.build(
            path,
            artifacts[kind],
            target="sm_89",
            compiler="nvrtc",
            nvrtc_home=os.environ["TENSOR_NVRTC_HOME"],
        )
    storage = get("fp8").storage((slots, heads, capacity, dim))
    with tensor.Device() as device:
        inputs = [
            device.from_numpy(k, dtype="bfloat16"),
            device.from_numpy(v),
            device.from_numpy(positions),
            device.from_numpy(active),
        ]
        caches = [
            device.from_numpy(np.full(storage.values.shape, 113, "uint8"))
            for _ in range(2)
        ]
        scales = [
            device.from_numpy(np.full(storage.metadata[0].shape, -9, "float32"))
            for _ in range(2)
        ]
        outputs = [device.empty(storage.values.shape, "bfloat16") for _ in range(2)]
        device.load(artifacts["kv_encode"]).launch(*inputs, *caches, *scales)
        device.load(artifacts["kv_decode"]).launch(*caches, *scales, *outputs)
        for source, cache, scale, output in zip((k, v), caches, scales, outputs):
            source = (
                torch.from_numpy(source)
                .bfloat16()
                .float()
                .reshape(rows, heads, dim // 128, 128)
            )
            expected_scale = source.abs().amax(-1).clamp_min(1e-12) / 448
            expected_bits = (
                (source / expected_scale[..., None])
                .to(torch.float8_e4m3fn)
                .view(torch.uint8)
                .reshape(rows, heads, dim)
            )
            expected_values = (
                (
                    expected_bits.view(torch.float8_e4m3fn)
                    .float()
                    .reshape(source.shape)
                    * expected_scale[..., None]
                )
                .reshape(rows, heads, dim)
                .bfloat16()
                .float()
            )
            data, metadata, decoded = (
                cache.to_numpy(),
                scale.to_numpy(),
                output.to_numpy(),
            )
            for row in range(rows):
                slot, position = row // chunk, positions[row]
                if active[row]:
                    np.testing.assert_array_equal(
                        data[slot, :, position], expected_bits[row].numpy()
                    )
                    np.testing.assert_array_equal(
                        metadata[slot, :, position], expected_scale[row].numpy()
                    )
                    np.testing.assert_array_equal(
                        decoded[slot, :, position], expected_values[row].numpy()
                    )
                else:
                    assert np.all(data[slot, :, position] == 113)
                    assert np.all(metadata[slot, :, position] == -9)


def test_scalar_codec_does_not_require_packed_attention_helpers(monkeypatch):
    from akas.quantization.kv.fp8 import layout
    from akas.quantization.kv import fp8
    from akas.ops.attention.cuda.quantized import make_kernel

    scalar = SimpleNamespace(writer=fp8.writer, decoder=fp8.decoder)
    codec = Format(
        "scalar_only", layout, (("cuda", "akas.quantization.kv.cuda.scalar_only"),)
    )
    monkeypatch.setitem(
        sys.modules,
        "akas.quantization.kv.scalar_only",
        SimpleNamespace(FORMAT=codec),
    )
    monkeypatch.setitem(
        sys.modules, "akas.quantization.kv.cuda.scalar_only", scalar
    )
    parameters = dict(r=1, capacity=64, kv_dtype="scalar_only")
    assert make_kernel("attention_partial", parameters) is not None
    with pytest.raises(ValueError, match="packed attention load"):
        make_kernel("attention_partial", dict(parameters, packed_loads=True))


@pytest.mark.parametrize("chunk", [0, -1, True])
def test_standalone_kernel_rejects_invalid_chunk(chunk):
    from akas.quantization.kv.fp8 import make_kernel

    with pytest.raises(ValueError, match="positive chunk"):
        make_kernel(
            "kv_encode", dict(slots=1, kv_heads=1, d=128, capacity=16, chunk=chunk)
        )
