"""Two-pass decode matches dense softmax, including idle partitions and tails."""

import os
from contextlib import ExitStack

import numpy as np
import pytest


@pytest.mark.parametrize("capacity,key_lanes,buffered", [
    (19, 8, False), (257, 4, False), (8193, 16, False), (131072, 8, False),
    (257, 8, True), (131072, 8, True), (257, 8, "packed"), (131072, 8, "packed"),
    (131072, 8, "packed-pad"), (19, 8, "packed64"), (257, 8, "packed64"),
    (131072, 8, "packed64"), (19, 8, "packed128"), (257, 8, "packed128"),
    (131072, 8, "packed128"), (257, 8, "packed64-chains"),
    (131072, 8, "packed64-chains")])
def test_decode_partition_matches_independent_reference(tmp_path, capacity, key_lanes, buffered):
    if os.environ.get("AKAS_LFM_MOE_WEBGPU") != "1":
        pytest.skip("opt-in WebGPU qualification")
    from akas.compilation.emit import build_artifact
    from akas.compilation.selection import source
    from akas.kernels.registry import validation
    from tensor.artifacts.format import read_artifact
    from tools.context_sweep import vulkan_device

    packed = buffered in ("packed", "packed-pad", "packed64", "packed128", "packed64-chains")
    p = dict(r=2, h=4, kh=1, d=64, capacity=capacity,
             splits=64 if packed else 256 if buffered else 128,
             optimized=True, provider="webgpu", kv_dtype="float16", unit_scales=True,
             granule=128, key_lanes=key_lanes, value_unroll=8, value_chains=4)
    if buffered == "packed-pad":
        p["cache_padding"] = 128
    if buffered in ("packed64", "packed64-chains"):
        p["key_rows"] = 64
    if buffered == "packed64-chains":
        p["dot_chains"] = 4
    if buffered == "packed128":
        p["key_rows"] = 128
    entry, artifact = tmp_path / "kernel.py", tmp_path / "kernel.tbin"
    entry.write_text(source("attention_adaptive_partial", p, contract="reference",
        provider="webgpu", target="webgpu-portable-v1",
        preferred="attention.webgpu.packed_decode" if packed else
        "attention.webgpu.buffered_decode" if buffered else "attention.webgpu.two_pass_decode").render())
    build_artifact(entry, artifact, provider="webgpu", target="webgpu-portable-v1")
    metadata, _ = read_artifact(artifact)
    case = validation("attention_adaptive_partial", p, metadata["arguments"])
    with vulkan_device() as device, ExitStack() as stack:
        buffers = []
        for arg in metadata["arguments"]:
            buffer = device.from_numpy(case.arguments[arg["name"]])
            stack.callback(buffer.release)
            buffers.append(buffer)
        kernel = stack.enter_context(device.load(artifact))
        kernel.launch(*buffers)
        for arg, buffer in zip(metadata["arguments"], buffers):
            np.testing.assert_allclose(buffer.to_numpy(), case.expected[arg["name"]],
                                       rtol=case.rtol, atol=case.atol)


def test_large_unaligned_capacity_keeps_the_incumbent():
    from akas.kernels.registry import select
    p = dict(r=1, h=4, kh=1, d=64, capacity=131071, splits=128,
             optimized=True, provider="webgpu", kv_dtype="float16", unit_scales=True)
    with pytest.raises(ValueError):
        select("attention_adaptive_partial", p, contract="reference", provider="webgpu",
               preferred="attention.webgpu.two_pass_decode")


@pytest.mark.parametrize("chunk,capacity,block_q,block_k", [(32,257,4,32), (32,8193,8,32), (8,257,4,16)])
def test_packed_prefill_matches_independent_reference(tmp_path, chunk, capacity, block_q, block_k):
    if os.environ.get("AKAS_LFM_MOE_WEBGPU") != "1":
        pytest.skip("opt-in WebGPU qualification")
    from akas.compilation.emit import build_artifact
    from akas.compilation.selection import source
    from akas.kernels.registry import validation
    from tensor.artifacts.format import read_artifact
    from tools.context_sweep import vulkan_device

    p = dict(r=chunk, chunk=chunk, h=4, kh=1, d=64, capacity=capacity,
             optimized=True, provider="webgpu", kv_dtype="float16", unit_scales=True,
             block_q=block_q, block_k=block_k, key_pad=0, shared_dtype="float16",
             dot_width=4, packed_shared=True)
    entry, artifact = tmp_path / "prefill.py", tmp_path / "prefill.tbin"
    entry.write_text(source("attention", p, contract="reference", provider="webgpu",
        target="webgpu-portable-v1", preferred="attention.webgpu.tiled_prefill").render())
    build_artifact(entry, artifact, provider="webgpu", target="webgpu-portable-v1")
    metadata, _ = read_artifact(artifact)
    case = validation("attention", p, metadata["arguments"])
    with vulkan_device() as device, ExitStack() as stack:
        buffers = []
        for arg in metadata["arguments"]:
            buffer = device.from_numpy(case.arguments[arg["name"]])
            stack.callback(buffer.release)
            buffers.append(buffer)
        kernel = stack.enter_context(device.load(artifact))
        kernel.launch(*buffers)
        for arg, buffer in zip(metadata["arguments"], buffers):
            np.testing.assert_allclose(buffer.to_numpy(), case.expected[arg["name"]],
                                       rtol=1e-4, atol=3e-6)


def test_cache_pitch_padding_retains_logical_capacity_and_precision():
    from akas.ops.attention.definition import cache_storage
    p = dict(r=2, kh=1, d=64, capacity=257, provider="webgpu",
             kv_dtype="float16", cache_padding=128)
    storage = cache_storage(p)
    assert storage.values.shape == (2, 1, 385, 64)
    assert storage.values.dtype == "float16" and p["capacity"] == 257
    assert storage.metadata[0].shape == (2, 1, 385, 1)
    for padding in (-32, 1, 4097, True):
        with pytest.raises(ValueError):
            cache_storage({**p, "cache_padding": padding})


@pytest.mark.parametrize("splits", [16, 64, 128, 512])
def test_stream_merge_matches_independent_reference(tmp_path, splits):
    if os.environ.get("AKAS_LFM_MOE_WEBGPU") != "1":
        pytest.skip("opt-in WebGPU qualification")
    from akas.compilation.emit import build_artifact
    from akas.compilation.selection import source
    from akas.kernels.registry import validation
    from tensor.artifacts.format import read_artifact
    from tools.context_sweep import vulkan_device

    p = dict(r=2, h=4, kh=1, d=64, capacity=257, splits=splits,
             optimized=True, provider="webgpu", kv_dtype="float16", unit_scales=True)
    entry, artifact = tmp_path / "merge.py", tmp_path / "merge.tbin"
    entry.write_text(source("attention_merge", p, contract="reference", provider="webgpu",
        target="webgpu-portable-v1", preferred="attention.webgpu.stream_merge").render())
    build_artifact(entry, artifact, provider="webgpu", target="webgpu-portable-v1")
    metadata, _ = read_artifact(artifact)
    case = validation("attention_merge", p, metadata["arguments"])
    with vulkan_device() as device, ExitStack() as stack:
        buffers = []
        for arg in metadata["arguments"]:
            buffer = device.from_numpy(case.arguments[arg["name"]])
            stack.callback(buffer.release)
            buffers.append(buffer)
        kernel = stack.enter_context(device.load(artifact))
        kernel.launch(*buffers)
        for arg, buffer in zip(metadata["arguments"], buffers):
            np.testing.assert_allclose(buffer.to_numpy(), case.expected[arg["name"]],
                                       rtol=case.rtol, atol=case.atol)
