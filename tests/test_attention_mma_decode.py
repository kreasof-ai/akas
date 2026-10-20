"""Tensor-core grouped decode against an independent online-softmax oracle."""

from contextlib import ExitStack
import os

import numpy as np
import pytest


def test_attention_declares_geometry_that_changes_its_reference():
    from akas.ops.attention.definition import IMPLEMENTATIONS

    for name in ("attention.cuda.mma_decode", "attention.cuda.pipelined"):
        implementation = next(i for i in IMPLEMENTATIONS if i.identifier == name)
        assert implementation.reference_keys == ("key_rows",)
    decode = next(
        i for i in IMPLEMENTATIONS if i.identifier == "attention.cuda.mma_decode"
    )
    assert not decode.accepts(
        dict(d=64, kv_dtype="float16", unit_scales=True, h=4, kh=1, capacity=8192)
    )

    params = dict(r=1, d=64, kv_dtype="fp8", h=4, kh=1, capacity=8192, optimized=True)
    assert decode.accepts(params)
    assert not decode.accepts({**params, "kv_dtype": "float32"})


@pytest.mark.parametrize("key_rows", [64, 128])
@pytest.mark.parametrize("kv_dtype", ["float16", "fp8"])
def test_cuda_grouped_mma_decode_padding_and_inactive(tmp_path, key_rows, kv_dtype):
    if os.environ.get("AKAS_LFM_MOE_CUDA") != "1":
        pytest.skip("opt-in native-half GQA qualification")
    import tensor
    from tensor.artifacts.format import read_artifact
    from akas.compilation.emit import build_artifact
    from akas.compilation.selection import source
    from akas.memory.transfer import write
    from akas.kernels.registry import validation

    p = dict(
        r=2,
        h=8,
        kh=2,
        d=64,
        capacity=8192,
        splits=16,
        chunk=1,
        kv_dtype=kv_dtype,
        unit_scales=kv_dtype == "float16",
        provider="cuda",
        optimized=True,
        key_rows=key_rows,
        stages=2,
        decode_precision="float16",
    )
    entry, artifact = tmp_path / "kernel.py", tmp_path / "kernel.tbin"
    entry.write_text(
        source(
            "attention_partial",
            p,
            contract="reference",
            provider="cuda",
            target="sm_89",
            preferred="attention.cuda.mma_decode",
        ).render()
    )
    build_artifact(entry, artifact, provider="cuda", target="sm_89")
    meta, _ = read_artifact(artifact)
    case = validation("attention_partial", p, meta["arguments"])
    with tensor.Device(provider="cuda") as device, ExitStack() as stack:
        buffers = {name: device.from_numpy(a) for name, a in case.arguments.items()}
        for buffer in buffers.values():
            stack.callback(buffer.release)
        module = stack.enter_context(device.load(artifact))
        arguments = [buffers[a["name"]] for a in meta["arguments"]]
        module.launch(*arguments)
        np.testing.assert_allclose(
            buffers["out"].to_numpy(),
            case.expected["out"],
            rtol=case.rtol,
            atol=case.atol,
        )
        for length in (32, 65, 8191, 8192):
            encoded_keys = case.arguments["kc"].copy()
            encoded_values = case.arguments["vc"].copy()
            if kv_dtype == "fp8":
                from akas.quantization._shared.e4m3 import decode_storage

                keys = (decode_storage(encoded_keys) * case.arguments["ks"]).astype(
                    "float16"
                )
                values = (decode_storage(encoded_values) * case.arguments["vs"]).astype(
                    "float16"
                )
                encoded_keys[:, :, length:] = 127
                encoded_values[:, :, length:] = 127
                for name in ("ks", "vs"):
                    scales = case.arguments[name].copy()
                    scales[:, :, length:] = np.nan
                    write(device, buffers[name], scales)
            else:
                keys, values = encoded_keys, encoded_values
            # Padding may be stale or nonfinite; zero-probability rows must
            # still produce finite output. The second request is disabled.
            keys[:, :, length:] = np.nan
            values[:, :, length:] = np.nan
            write(device, buffers["kc"], encoded_keys)
            write(device, buffers["vc"], encoded_values)
            write(
                device,
                buffers["positions"],
                np.array([length - 1, length - 1], dtype="int32"),
            )
            write(device, buffers["enabled"], np.array([1, 0], dtype="int32"))
            expected = np.zeros((2, 8, 16, 66), dtype="float64")
            expected[..., 64] = -np.inf
            span = ((length + key_rows - 1) // key_rows + 15) // 16 * key_rows
            queries = (
                case.arguments["q"]
                .reshape(2, 8, 64)
                .astype("float16")
                .astype("float64")
            )
            for head in range(8):
                for part in range(16):
                    maximum, normalizer = -np.inf, 0.0
                    acc = np.zeros(64)
                    for start in range(
                        part * span, min(length, (part + 1) * span), key_rows
                    ):
                        end = min(length, start + key_rows)
                        score = (
                            keys[0, head // 4, start:end].astype("float64")
                            @ queries[0, head]
                            / 8
                        )
                        next_max = max(maximum, score.max())
                        factor = (
                            np.exp(maximum - next_max) if np.isfinite(maximum) else 0.0
                        )
                        probability = np.exp(score - next_max)
                        acc = acc * factor + probability.astype("float16").astype(
                            "float64"
                        ) @ values[0, head // 4, start:end].astype("float64")
                        normalizer = normalizer * factor + probability.sum()
                        maximum = next_max
                    expected[0, head, part, :64] = acc
                    expected[0, head, part, 64:] = [maximum, normalizer]
            module.launch(*arguments)
            actual = buffers["out"].to_numpy().reshape(expected.shape)
            np.testing.assert_allclose(actual, expected, rtol=0.002, atol=0.002)


def test_tuning_rebuilds_mma_reference_for_key_tile(tmp_path, monkeypatch):
    if os.environ.get("AKAS_LFM_MOE_CUDA") != "1":
        pytest.skip("opt-in CUDA tuning qualification")
    from dataclasses import replace
    from akas.graph import Graph
    from akas.compilation.compile import produce
    from akas.compilation.tuning import SearchSettings
    from akas.kernels import registry
    from akas.ops.attention.definition import IMPLEMENTATIONS

    choice = next(
        i for i in IMPLEMENTATIONS if i.identifier == "attention.cuda.mma_decode"
    )
    choice = replace(choice, space=(("key_rows", (64, 128)),))
    graph = Graph(contract="reference")
    p = dict(
        r=1,
        h=4,
        kh=1,
        d=64,
        capacity=8192,
        splits=16,
        chunk=1,
        kv_dtype="float16",
        unit_scales=True,
        provider="cuda",
        optimized=True,
    )
    layout = [
        ("q", (1, 256), "float32"),
        ("kc", (1, 1, 8192, 64), "float16"),
        ("vc", (1, 1, 8192, 64), "float16"),
        ("ks", (1, 1, 8192, 1), "float32"),
        ("vs", (1, 1, 8192, 1), "float32"),
        ("out", (4 * 16 * 66,), "float32"),
        ("positions", (1,), "int32"),
        ("enabled", (1,), "int32"),
    ]
    args = [
        graph.buffer(n, s, d, role="output" if n == "out" else "input")
        for n, s, d in layout
    ]
    graph("attention_partial", p, *args)
    original = registry.validation
    tiles = []

    def counted(kind, parameters, signature):
        tiles.append(parameters["key_rows"])
        return original(kind, parameters, signature)

    monkeypatch.setattr(registry, "validation", counted)
    result = produce(
        graph,
        tmp_path,
        provider="cuda",
        catalog=(choice,),
        search_settings=SearchSettings(
            budget=2, width=1, warmup=0, samples=1, repeats=1
        ),
    )
    assert set(tiles) == {64, 128}
    assert len(tiles) == 2
    assert all(
        c["status"] == "passed" for c in result["tuning"]["operations"][0]["candidates"]
    )


@pytest.mark.parametrize("threads", [128, 256])
def test_cuda_packed_prefill_masks_inactive_query_tiles(tmp_path, threads):
    if os.environ.get("AKAS_LFM_MOE_CUDA") != "1":
        pytest.skip("opt-in packed prefill qualification")
    import tensor
    from tensor.artifacts.format import read_artifact
    from akas.compilation.emit import build_artifact
    from akas.compilation.selection import source
    from akas.memory.transfer import write
    from akas.kernels.registry import validation

    p = dict(
        r=66,
        chunk=33,
        h=8,
        kh=2,
        d=64,
        capacity=4096,
        kv_dtype="fp8",
        provider="cuda",
        optimized=True,
        threads=threads,
        prefill_precision="float16",
        key_rows=64,
    )
    entry, artifact = tmp_path / "kernel.py", tmp_path / "kernel.tbin"
    entry.write_text(
        source(
            "attention",
            p,
            contract="reference",
            provider="cuda",
            target="sm_89",
            preferred="attention.cuda.packed_prefill",
        ).render()
    )
    build_artifact(entry, artifact, provider="cuda", target="sm_89")
    metadata, _ = read_artifact(artifact)
    case = validation("attention", p, metadata["arguments"])
    # The last query tile of slot 1 is inactive. Its future cache location is
    # poisoned; only the live queries may determine the staged cache length.
    for name in ("kc", "vc"):
        case.arguments[name][-1, :, -1] = 127
    for name in ("ks", "vs"):
        case.arguments[name][-1, :, -1] = np.nan
    with tensor.Device(provider="cuda") as device, ExitStack() as stack:
        buffers = {
            name: device.from_numpy(array) for name, array in case.arguments.items()
        }
        for buffer in buffers.values():
            stack.callback(buffer.release)
        module = stack.enter_context(device.load(artifact))
        args = [buffers[a["name"]] for a in metadata["arguments"]]
        module.launch(*args)
        np.testing.assert_allclose(
            buffers["out"].to_numpy(),
            case.expected["out"],
            rtol=case.rtol,
            atol=case.atol,
        )
        write(device, buffers["enabled"], np.zeros(66, dtype="int32"))
        module.launch(*args)
        np.testing.assert_array_equal(buffers["out"].to_numpy(), 0)
