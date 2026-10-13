"""LFM token phases share the public executor and preserve mapped request state."""

import os
from functools import partial
from types import SimpleNamespace

import numpy as np
import pytest

from support.diagnostic import fixture
from akas.models.lfm2.weights import ReferenceCheckpoint
from akas.models.lfm2.model import compose_program


def test_lfm_phase_bindings_and_pool_layout(tmp_path):
    checkpoint = ReferenceCheckpoint(fixture(tmp_path / "model.gguf"), tokenizer=False)
    for chunk in (1, 4):
        graph = compose_program(checkpoint, rows=2, chunk=chunk,
                                max_requests=4, max_context=32)
        assert graph.request["inputs"]["tokens"] == ("tokens" if chunk == 1 else "prefill_tokens")
        assert graph.request["prediction"] == "sampled"
        assert graph.buffers["positions"].shape == (4,)
        assert graph.buffers["blk.0.conv_state"].shape[0] == 4
        assert graph.buffers["blk.1.kc"].shape[0] == 4
        assert set(graph.request["reset"]) == {"positions", "blk.0.conv_state"}


@pytest.mark.parametrize("provider", ["cuda", "webgpu"])
@pytest.mark.parametrize("kv_dtype", ["float16", "fp8"])
def test_lfm_common_executor_ragged_reordered_and_reused_requests(tmp_path, provider, kv_dtype):
    if os.environ.get("AKAS_LFM_REFERENCE_" + provider.upper()) != "1":
        pytest.skip("enable hardware program qualification")
    import tensor
    from akas.compilation.compile import compile_program
    from akas.compilation.tuning import SearchSettings
    from akas.execution.executor import Executor
    from support.oracle import OptimizedOracle

    path = fixture(tmp_path / "model.gguf")
    checkpoint = ReferenceCheckpoint(path, tokenizer=False)
    destination = tmp_path / provider
    compile_program(checkpoint, destination,
                    definition=partial(compose_program, kv_dtype=kv_dtype),
                    geometries=[(2, 1), (2, 4)], max_requests=4, max_context=32,
                    provider=provider, upload_options={"matrix_dtype": "float16"},
                    search_settings=SearchSettings(budget=2, samples=1, repeats=1))
    with tensor.Device(provider=provider) as device:
        with Executor.from_program(checkpoint, destination, device) as executor:
            slots = [executor.admit_at(3), executor.admit_at(1)]
            for sequences, chunk in (([[1, 2, 3, 4], [5, 6]], 4), ([[7], [8]], 1), ([[9]], 1)):
                active = slots[:len(sequences)]
                frame = executor.batches.select(len(active), chunk)
                expected = []
                for slot, sequence in zip(active, sequences):
                    oracle = OptimizedOracle(path, provider=provider, batch=2, chunk=chunk,
                                             context=32, kv_dtype=kv_dtype)
                    oracle.load_state(SimpleNamespace(positions=executor.positions,
                                                      buffers=executor.batches.states), slot)
                    expected.append(oracle.forward(sequence, prefill=chunk > 1))
                untouched = {name: value.to_numpy()[[0, 2]].copy()
                             for name, value in executor.batches.states.items()}
                executor.run_batch(active, sequences, chunk=chunk)
                actual = frame.forward.buffers["logits"].to_numpy().reshape(2, -1)
                for row, reference in enumerate(expected):
                    error = np.sqrt(np.mean((actual[row] - reference) ** 2))
                    magnitude = max(float(np.sqrt(np.mean(reference ** 2))), 1e-6)
                    assert error / magnitude < 0.02
                for name, reference in untouched.items():
                    np.testing.assert_array_equal(executor.batches.states[name].to_numpy()[[0, 2]], reference)
                np.testing.assert_array_equal(executor.batches.states["positions"].to_numpy(), executor.positions)
                slots.reverse()
            for slot in slots:
                executor.release(slot)
                assert executor.admit_at(slot) == slot
                assert executor.positions[slot] == 0
                for name in ("positions", "blk.0.conv_state"):
                    np.testing.assert_array_equal(executor.batches.states[name].to_numpy()[slot], 0)
                executor.release(slot)
        assert not device._buffers


@pytest.mark.parametrize("provider", ["cuda", "webgpu"])
def test_lfm_packed_program_preserves_weight_bytes_and_mapped_state(tmp_path, provider):
    if os.environ.get("AKAS_LFM_REFERENCE_" + provider.upper()) != "1":
        pytest.skip("enable hardware program qualification")
    import tensor
    from akas.models.lfm2.weights import LFM2Checkpoint
    from akas.models.lfm2.model import LFM2
    from akas.compilation.compile import compile_program
    from akas.compilation.tuning import SearchSettings
    from akas.execution.executor import Executor

    path = fixture(tmp_path / "model.gguf")
    checkpoint = LFM2Checkpoint(path, tokenizer=False)
    reference = ReferenceCheckpoint(path, tokenizer=False)
    settings = SearchSettings(budget=2, samples=1, repeats=1)
    for name, loaded in (("packed", checkpoint), ("floating", reference)):
        compile_program(loaded, tmp_path / name, definition=compose_program,
                        geometries=[(2, 1), (2, 4)], max_requests=4, max_context=32,
                        provider=provider, search_settings=settings)
    with tensor.Device(provider=provider) as device:
        with LFM2(path, tmp_path / "packed", device, checkpoint_options={"tokenizer": False}) as packed, \
             Executor.from_program(reference, tmp_path / "floating", device) as floating:
            for engine in (packed, floating):
                engine.admit_at(3)
                engine.admit_at(1)
            for slots, sequences, chunk in (([3, 1], [[1, 2, 3, 4], [5, 6]], 4),
                                             ([1, 3], [[7], [8]], 1),
                                             ([3], [[9]], 1)):
                for engine in (packed, floating):
                    engine.run_batch(slots, sequences, chunk=chunk)
                actual = packed.batches.select(len(slots), chunk).forward.buffers["logits"].to_numpy()
                expected = floating.batches.select(len(slots), chunk).forward.buffers["logits"].to_numpy()
                for row in range(len(slots)):
                    a = actual.reshape(2, -1)[row]
                    b = expected.reshape(2, -1)[row]
                    assert np.linalg.norm(a-b) / max(np.linalg.norm(b), 1e-6) < 0.02
                for engine in (packed, floating):
                    np.testing.assert_array_equal(engine.batches.states["positions"].to_numpy(), engine.positions)
                    for name, state in engine.batches.states.items():
                        if name.endswith("conv_state"):
                            np.testing.assert_array_equal(state.to_numpy()[[0, 2]], 0)
            for name, value in packed.weights.items():
                raw = checkpoint.gguf.packed(name)
                np.testing.assert_array_equal(value.to_numpy().view("uint8"), raw)
        assert not device._buffers


def test_public_definitions_share_executor_and_loader_dispatch(tmp_path):
    from support.checkpoint import export as hf_fixture
    from akas.models.definition import Definition
    from akas.models.lfm2.model import LFM2, LFM2Batched, LFM2Reference
    from akas.models.lfm2.weights import LFM2Checkpoint

    path = fixture(tmp_path / "model.gguf")
    directory = hf_fixture(path, tmp_path / "hf")
    for model in (LFM2,LFM2Batched,LFM2Reference):
        assert issubclass(model,Definition)
        assert model.compilation_profile == "program"
    gguf = LFM2Checkpoint(path,tokenizer=False)
    safetensors = LFM2Checkpoint(directory,tokenizer=False)
    assert gguf.config == safetensors.config
    assert hasattr(gguf,"graph_weights")
    assert not hasattr(safetensors,"graph_weights")
