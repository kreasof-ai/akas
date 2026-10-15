"""GGUF dense/MoE mapping, storage preservation and numerical consumers."""

from math import prod
import os

import numpy as np
import pytest

from akas.checkpoints.gguf import GGUFError, dequantize
from akas.graph import Graph
from akas.models.qwen35 import Qwen35Dense, Qwen35MoE
from support.qwen_gguf import fixture


@pytest.mark.parametrize("moe", [False, True])
@pytest.mark.parametrize("output", [False, True])
def test_checkpoint_dispatch_mapping_and_packed_graph(tmp_path, moe, output):
    from akas.quantization.weights import Weight

    declaration = Qwen35MoE if moe else Qwen35Dense
    checkpoint = declaration.checkpoint_type(
        fixture(tmp_path / "model.gguf", moe=moe, output=output)
    )
    c = checkpoint.config
    assert c.width == 256 and c.layers == ("linear_attention", "full_attention")
    if moe:
        assert c.experts == 4 and c.top_k == 2
    for chunk in (1, 4):
        graph = declaration.compose(
            checkpoint, rows=2, chunk=chunk, max_requests=4, max_context=64
        ).graph
        weights = checkpoint.graph_weights(Graph(contract="bf16"), provider="cuda")
        projected = weights["model.language_model.layers.0.linear_attn.input.weight"]
        assert isinstance(projected, Weight) and len(projected.parts) == 4
        assert [part["options"]["encoding"] for part in projected.parts] == [
            13,
            12,
            8,
            8,
        ]
        kinds = {call.kind for call in graph.calls}
        assert {"quantized_projection", "quantized_embedding"} <= kinds
        if moe:
            assert "quantized_experts" in kinds
        matrices = [
            spec
            for name, spec in graph.buffers.items()
            if name in checkpoint.used and name not in checkpoint.auxiliary
        ]
        assert any(spec.dtype == "uint8" for spec in matrices)
        assert sum(prod(spec.shape) for spec in matrices) < 8_000_000
    assert checkpoint.head_name == (
        "lm_head.weight" if output else "model.language_model.embed_tokens.weight"
    )
    np.testing.assert_array_equal(
        checkpoint.read("model.language_model.norm.weight"), np.zeros(256)
    )
    np.testing.assert_allclose(
        checkpoint.read("model.language_model.layers.0.linear_attn.A_log"), np.log(0.5)
    )


@pytest.mark.parametrize("moe", [False, True])
def test_wrong_model_family_and_incomplete_inventory_rejected(tmp_path, moe):
    path = fixture(tmp_path / "model.gguf", moe=moe)
    wrong = Qwen35Dense if moe else Qwen35MoE
    with pytest.raises(GGUFError, match="architecture"):
        wrong.checkpoint_type(path)
    path = fixture(
        path, moe=moe, mutate=lambda metadata, tensors: tensors.pop("blk.0.ssm_a")
    )
    with pytest.raises(GGUFError, match="missing.*ssm_a"):
        (Qwen35MoE if moe else Qwen35Dense).checkpoint_type(path)


def test_q5_k_has_independent_high_bits_scales_and_minima():
    # Encode every group with an independent 5-bit sequence and simple scales.
    raw = np.zeros(176, dtype="uint8")
    raw[:2] = np.array([0.5], dtype="<f2").view("uint8")
    raw[2:4] = np.array([0.25], dtype="<f2").view("uint8")
    scales = np.arange(1, 9, dtype="uint8")
    minima = np.arange(9, 17, dtype="uint8")
    raw[4:8] = scales[:4] | (scales[4:] >> 4) << 6
    raw[8:12] = minima[:4] | (minima[4:] >> 4) << 6
    raw[12:16] = (scales[4:] & 15) | (minima[4:] & 15) << 4
    codes = (np.arange(256).reshape(8, 32) + np.arange(8)[:, None] * 3) % 32
    for group in range(8):
        raw[16:48] |= (codes[group] >> 4).astype("uint8") << group
        raw[48 + group // 2 * 32 : 80 + group // 2 * 32] |= (codes[group] & 15).astype(
            "uint8"
        ) << (4 * (group % 2))
    expected = 0.5 * scales[:, None] * codes - 0.25 * minima[:, None]
    np.testing.assert_array_equal(dequantize(raw, 13), expected.reshape(-1))


def test_grouped_value_order_is_restored_for_rows_columns_and_vectors():
    # Use four V heads per K head, so the permutation is not the identity.
    from akas.quantization._shared.access import restore_order

    source = np.arange(8 * 3).reshape(8, 3)
    tiled = source.reshape(2, 4, 3).transpose(1, 0, 2).reshape(8, 3)
    np.testing.assert_array_equal(restore_order(tiled, 0, (0, 2, 4, 1)), source)
    np.testing.assert_array_equal(restore_order(tiled.T, 1, (0, 2, 4, 1)), source.T)


def test_upload_preserves_matrix_bytes_and_releases_partial_allocations(tmp_path):
    from types import SimpleNamespace

    checkpoint = Qwen35MoE.checkpoint_type(fixture(tmp_path / "model.gguf", moe=True))
    read = checkpoint.read

    def auxiliaries_only(name):
        assert name in checkpoint.auxiliary, (
            "matrix decoding must stay fused in consumers"
        )
        return read(name)

    checkpoint.read = auxiliaries_only

    class Device:
        info = {"provider": "cuda"}

        def __init__(self, fail=False):
            self.buffers = []
            self.fail = fail

        def from_numpy(self, values, **options):
            if self.fail and len(self.buffers) == 2:
                raise RuntimeError("synthetic transfer failure")
            buffer = SimpleNamespace(
                values=values.copy(), options=options, released=False
            )

            def release():
                buffer.released = True

            buffer.release = release
            self.buffers.append(buffer)
            return buffer

    device = Device()
    weights = checkpoint.upload(device)
    for name, buffer in weights.items():
        if name not in checkpoint.auxiliary:
            np.testing.assert_array_equal(
                buffer.values.view("uint8").reshape(-1), checkpoint.gguf.packed(name)
            )
    failed = Device(fail=True)
    with pytest.raises(RuntimeError, match="synthetic transfer failure"):
        checkpoint.upload(failed)
    assert len(failed.buffers) == 2 and all(
        buffer.released for buffer in failed.buffers
    )


def consumer_cases():
    yield (
        "quantized_projection",
        dict(
            r=3,
            k=96,
            o=7,
            block_k=64,
            input_dtype="float32",
            weight_parts=({"rows": 7, "format": "ggml", "options": {"encoding": 8}},),
        ),
    )
    for rows in (1, 5):
        for count in (1, 2, 3, 4):
            parts = tuple(
                dict(
                    rows=24 if index == 0 else 8,
                    format="ggml",
                    options={"encoding": (13, 12, 8, 14)[index]},
                    row_order=(0, 2, 3, 4) if index == 0 else None,
                    column_order=(0, 2, 2, 64) if index == 0 else None,
                )
                for index in range(count)
            )
            yield (
                "quantized_projection",
                dict(
                    r=rows,
                    k=256,
                    o=sum(part["rows"] for part in parts),
                    input_dtype="float32",
                    weight_parts=parts,
                ),
            )
    yield (
        "quantized_embedding",
        dict(
            r=3,
            c=256,
            vocab=7,
            output_dtype="float32",
            weight_parts=({"rows": 7, "format": "ggml", "options": {"encoding": 13}},),
        ),
    )
    yield "quantized_routes", dict(r=5, top=2, experts=4)
    yield (
        "swiglu_experts",
        dict(
            r=2,
            top=2,
            c=16,
            expert_rows=True,
            round_inputs=True,
            output_dtype="bfloat16",
        ),
    )
    for rows in (1, 5):
        for routed in (False, True):
            yield (
                "quantized_experts",
                dict(
                    r=rows,
                    top=2,
                    experts=4,
                    o=64,
                    k=256,
                    routed_input=routed,
                    weight_parts=(
                        {"rows": 256, "format": "ggml", "options": {"encoding": 13}},
                    ),
                ),
            )


CONSUMERS = list(consumer_cases())


@pytest.mark.parametrize("provider", ["cuda", "webgpu"])
@pytest.mark.parametrize("kind,parameters", CONSUMERS)
def test_encoded_consumer_native_qualification(tmp_path, provider, kind, parameters):
    from contextlib import ExitStack

    enabled = (
        os.environ.get("AKAS_QWEN_GGUF_CUDA") == "1"
        if provider == "cuda"
        else os.environ.get("AKAS_COMPILE_WEBGPU") == "1"
    )
    if not enabled:
        pytest.skip("opt-in GGUF CUDA execution or WebGPU shader compilation")
    if provider == "webgpu" and kind in ("quantized_experts", "swiglu_experts"):
        pytest.skip("Qwen's native BF16 expert graph currently requires CUDA")
    import tensor
    from tensor.artifacts.format import read_artifact
    from akas.compilation.emit import build_artifact
    from akas.compilation.selection import source
    from akas.kernels.registry import validation

    entry = tmp_path / "kernel.py"
    artifact = tmp_path / "kernel.tbin"
    target = "sm_89" if provider == "cuda" else "webgpu-portable-v1"
    entry.write_text(
        source(
            kind, parameters, contract="bf16", provider=provider, target=target
        ).render()
    )
    build_artifact(entry, artifact, provider=provider, target=target)
    meta, files = read_artifact(artifact)
    if provider == "webgpu":
        assert any(name.endswith(".wgsl") for name in files)
        return
    case = validation(kind, parameters, meta["arguments"])
    with tensor.Device() as device, ExitStack() as stack:
        buffers = []
        for argument in meta["arguments"]:
            buffer = device.from_numpy(
                case.arguments[argument["name"]], dtype=argument["dtype"]
            )
            stack.callback(buffer.release)
            buffers.append(buffer)
        kernel = stack.enter_context(device.load(artifact))
        kernel.launch(*buffers)
        for argument, buffer in zip(meta["arguments"], buffers):
            name = argument["name"]
            np.testing.assert_allclose(
                buffer.to_numpy(), case.expected[name], rtol=case.rtol, atol=case.atol
            )


@pytest.mark.parametrize("moe", [False, True])
def test_small_gguf_model_compiles_tunes_and_runs_reordered_slots(tmp_path, moe):
    if os.environ.get("AKAS_QWEN_GGUF_CUDA") != "1":
        pytest.skip("opt-in full GGUF model CUDA qualification")
    import tensor
    from akas.compilation import compile_checkpoint
    from akas.compilation.tuning import SearchSettings
    from akas.benchmarking import Case, run
    from akas.benchmarking.session import Session

    if os.environ.get("AKAS_QWEN_GGUF_BUILD"):
        from pathlib import Path

        tmp_path = Path(os.environ["AKAS_QWEN_GGUF_BUILD"]) / (
            "moe" if moe else "dense"
        )
        tmp_path.mkdir(parents=True, exist_ok=True)

    declaration = Qwen35MoE if moe else Qwen35Dense
    path = fixture(tmp_path / "model.gguf", moe=moe)
    directory = tmp_path / "program"
    compile_checkpoint(
        declaration,
        path,
        directory,
        geometries=[(1, 1), (2, 1), (2, 4)],
        max_requests=3,
        max_context=64,
        target="sm_89",
        search_settings=SearchSettings(
            budget=1, width=1, warmup=0, samples=1, repeats=1
        ),
    )
    with tensor.Device() as device, declaration(path, directory, device) as model:
        slots = [model.admit_at(2), model.admit_at(0)]
        predicted = model.prefill(slots, [[1, 2, 3], [4, 5]], chunk=4)
        assert all(0 <= token < 64 for token in predicted)
        model.decode(slots[::-1], predicted[::-1])
        assert model.positions.tolist() == [3, 0, 4]
        for slot in slots:
            model.release(slot)
        with Session(model, batch=2, chunk=4) as session:
            result = run(session, Case(2, 3, generated=2), repeats=1, warmup=0)
            assert result["status"] == "ok"
