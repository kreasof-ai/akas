"""Optimized floating execution preserves masks, tails and loader equivalence."""

import os
from pathlib import Path
import numpy as np
import pytest
from akas.models.lfm2.model import LFM2Batched
from akas.models.lfm2.weights import ReferenceCheckpoint
from akas.compilation.compile import plan
from benchmarks.lfm2.diagnostic import fixture
from examples.lfm2.checkpoint import export


@pytest.fixture(scope="module")
def checkpoints(tmp_path_factory):
    root = tmp_path_factory.mktemp("lfm-floating")
    gguf = fixture(root / "model.gguf")
    return gguf, export(gguf, root / "hf")


def test_optimized_selection_and_scratch_lifetimes(checkpoints):
    from akas.compilation.buffers import scratch_slots

    config = ReferenceCheckpoint(checkpoints[0], tokenizer=False).config
    forward = LFM2Batched.definition(config, batch=4, capacity=256, chunk=32)
    bundle = plan(forward)
    assert {k.source.args[3]["module"] for k in bundle.kernels if k.kind == "linear"} == {
        "akas.ops.linear.cuda.floating"
    }
    assert {k.source.args[3]["module"] for k in bundle.kernels if k.kind == "attention"} == {
        "akas.ops.attention.cuda.floating"
    }
    assert forward.buffers["blk.1.kc"].shape[0] == 4
    groups = scratch_slots(forward)
    assert any(len(names) > 1 for names in groups)
    for names in groups:
        previous_end = -1
        for name in names:
            uses = [
                i
                for i, call in enumerate(forward.calls)
                if name in (*call.arguments, *call.bindings.values())
            ]
            assert uses[0] > previous_end
            previous_end = uses[-1]


@pytest.mark.parametrize("provider", ["cuda", "webgpu"])
def test_default_prefill_chunk_fits_a_small_context(
    checkpoints, tmp_path, monkeypatch, provider
):
    from akas.compilation import compile_checkpoint
    from akas.compilation import compile as session

    def compose_chunk(checkpoint, out, *, definition, **options):
        return definition(
            checkpoint.config,
            batch=options["batch"],
            capacity=options["capacity"],
            provider=options["provider"],
            chunk=options["prefill_chunk"],
        )

    monkeypatch.setattr(session, "compile_model", compose_chunk)
    forward = compile_checkpoint(
        LFM2Batched,
        checkpoints[0],
        tmp_path,
        profile="floating",
        checkpoint_options={"tokenizer": False},
        provider=provider,
        batch=4,
        capacity=7,
    )
    assert forward.buffers["prefill_tokens"].shape == (28,)
    assert plan(forward, provider=provider).kernels


@pytest.mark.parametrize(
    "batch", [1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 63, 64, 65, 255, 256, 257]
)
@pytest.mark.parametrize("provider", ["cuda", "webgpu"])
def test_dispatch_boundaries_keep_optimized_projections(checkpoints, batch, provider):
    config = ReferenceCheckpoint(checkpoints[0], tokenizer=False).config
    for chunk in (1, 32):
        forward = LFM2Batched.definition(
            config, batch=batch, capacity=256, chunk=chunk, provider=provider
        )
        bundle = plan(forward, provider=provider)
        expected = f"akas.ops.linear.{provider}.floating"
        assert {
            k.source.args[3]["module"] for k in bundle.kernels if k.kind in ("linear", "ffn")
        } == {expected}
        if chunk > 1:
            # The stateful mixer retains every token; its terminal pointwise
            # suffix computes only each slot's selected final row.
            assert forward.buffers["blk.1.kc"].shape[0] == batch
            assert any(call.kind == "chunk_last" for call in forward.calls)


@pytest.mark.parametrize("kv_dtype", ["float16", "fp8"])
@pytest.mark.parametrize("provider", ["cuda", "webgpu"])
def test_chunked_mixed_tails_decode_and_partial_reset(
    checkpoints, tmp_path, kv_dtype, provider
):
    if os.environ.get("AKAS_LFM_REFERENCE_" + provider.upper()) != "1":
        pytest.skip("enable provider qualification")
    import tensor
    from akas.compilation import compile_checkpoint
    from examples.lfm2.oracle import Oracle

    bundle = (
        Path(os.environ.get("AKAS_LFM_FLOATING_BUILD", tmp_path))
        / f"{provider}-{kv_dtype}"
    )
    compile_checkpoint(
        LFM2Batched,
        checkpoints[0],
        bundle,
        profile="reference",
        checkpoint_options={"tokenizer": False},
        provider=provider,
        batch=4,
        capacity=256,
        prefill_chunk=32,
        kv_dtype=kv_dtype,
    )
    with tensor.Device(provider=provider) as device:
        for checkpoint in checkpoints:
            with LFM2Batched(
                checkpoint, bundle, device, tokenizer=False, graphs=True
            ) as model:
                prompts = [
                    np.random.default_rng(slot).integers(
                        1, model.config.vocab, length, dtype=np.int32
                    )
                    for slot, length in enumerate((1, 31, 33, 129))
                ]
                expected = []
                oracles = [
                    Oracle(checkpoint, context=256, kv_dtype=kv_dtype) for _ in prompts
                ]
                for oracle, prompt in zip(oracles, prompts):
                    expected.append(oracle.forward(prompt))
                actual = model.prefill(prompts)
                for got, want in zip(actual, expected):
                    assert np.isfinite(got).all()
                    assert np.linalg.norm(got - want) / np.linalg.norm(want) < 0.01
                model.step([1, 2, 3, 4])
                for got, oracle, token in zip(model.logits(), oracles, [1, 2, 3, 4]):
                    want = oracle.forward([token])
                    assert np.linalg.norm(got - want) / np.linalg.norm(want) < 0.01
                np.testing.assert_array_equal(model.positions, [2, 32, 34, 130])
                untouched = {
                    name: model.buffers[name].to_numpy()[1:].copy()
                    for name, spec in model.forward.definition.buffers.items()
                    if spec.role == "state"
                }
                model.reset([0])
                oracles[0].reset()
                actual = model.step([5, 1, 1, 1], active=[1, 0, 0, 0])
                want = oracles[0].forward([5])
                assert np.linalg.norm(actual[0] - want) / np.linalg.norm(want) < 0.01
                # Partial reset keeps the other slots' persistent positions.
                np.testing.assert_array_equal(model.positions, [1, 32, 34, 130])
                for name, values in untouched.items():
                    np.testing.assert_array_equal(
                        model.buffers[name].to_numpy()[1:], values
                    )


@pytest.mark.parametrize("algorithm", ["auto", "mma"])
@pytest.mark.parametrize(
    "rows,k", [(3, 256), (8, 320), (9, 256), (31, 320), (32, 256), (33, 320)]
)
@pytest.mark.parametrize("kind", ["linear", "ffn"])
@pytest.mark.parametrize("provider", ["cuda", "webgpu"])
def test_floating_projection_preserves_partitioned_output(
    tmp_path, rows, k, kind, provider, algorithm
):
    if os.environ.get("AKAS_LFM_REFERENCE_" + provider.upper()) != "1":
        pytest.skip("enable provider qualification")
    import tensor
    from akas.compilation.emit import build_artifact
    from akas.compilation.emit import export_kernel

    p = dict(
        r=rows,
        k=k,
        o=64,
        optimized=True,
        weight_dtype="float16",
        matrix_output=True,
        output_stride=128,
        output_offset=64,
        algorithm=algorithm,
    )
    source, artifact = tmp_path / "kernel.py", tmp_path / "kernel.tbin"
    source.write_text(
        export_kernel(f"akas.ops.linear.{provider}.floating", "make_kernel", kind, p)
    )
    rng = np.random.default_rng(43)
    x = rng.normal(size=(rows, k)).astype("float32")
    w = rng.normal(0, 0.02, size=(64, k)).astype("float16")
    half = algorithm == "mma" or (
        algorithm == "auto" and provider == "cuda" and rows > 31
    )
    operand = x.astype("float16").astype("float32") if half else x
    expected = operand @ w.astype("float32").T
    arrays = [x.reshape(-1), w.reshape(-1)]
    if kind == "ffn":
        w2 = rng.normal(0, 0.02, size=w.shape).astype("float16")
        arrays.append(w2.reshape(-1))
        expected = (
            expected / (1 + np.exp(-expected)) * (operand @ w2.astype("float32").T)
        )
    arrays.append(np.zeros((rows, 128), "float32"))
    with tensor.Device(provider=provider) as device:
        build_artifact(source, artifact, target=device.info["arch"], provider=provider)
        buffers = [device.from_numpy(a) for a in arrays]
        try:
            with device.load(artifact) as kernel:
                kernel.launch(*buffers)
                device.synchronize()
            actual = buffers[-1].to_numpy()
            np.testing.assert_array_equal(actual[:, :64], 0)
            np.testing.assert_allclose(actual[:, 64:], expected, atol=1e-6, rtol=1e-5)
        finally:
            for buffer in buffers:
                buffer.release()
