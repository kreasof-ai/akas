"""A decode import must retain the real prefill's identity and storage ABI."""

import copy
import json
from types import SimpleNamespace
import numpy as np
import pytest
from akas.benchmarking.history import digest
from akas.benchmarking.workload import prompt_digest
from tools.lfm3_state_cache import load
from tools.lfm3_vulkan_benchmark import PROFILE
from tools.model_benchmark import prompts


@pytest.fixture
def cache(tmp_path):
    program = dict(
        config={"head_dim": 128},
        max_requests=1,
        max_context=256,
        provider="webgpu",
        model_options={"activation_quantization": "mixed", "decode_splits": 128},
    )
    producer = tmp_path / "program"
    producer.mkdir()
    (producer / "program.json").write_text(json.dumps(program))
    directory = tmp_path / "state"
    directory.mkdir()
    values = np.array([[1, 2]], dtype="float16")
    np.save(directory / "0.npy", values, allow_pickle=False)
    metadata = dict(
        checkpoint_sha256=PROFILE.checkpoint_sha256,
        prompt_sha256=prompt_digest(prompts(2, PROFILE.vocab)),
        context=256,
        prefix=2,
        positions=[2],
        first=[5],
        producer_program=dict(
            path=str(producer), sha256=digest(producer / "program.json")
        ),
        files={
            "kv": dict(
                file="0.npy",
                sha256=digest(directory / "0.npy"),
                shape=[1, 2],
                dtype="float16",
            )
        },
    )
    (directory / "state.json").write_text(json.dumps(metadata))
    record = dict(
        id="producer",
        status="passed",
        metadata=dict(
            runtime="akas",
            protocol="real-prefill-state-restored-wall-decode-v1",
            checkpoint_sha256=PROFILE.checkpoint_sha256,
            max_context=256,
        ),
        results=[
            dict(
                case={"prompt": 2},
                prompt_sha256=metadata["prompt_sha256"],
                program_sha256=metadata["producer_program"]["sha256"],
                output_ids=[5],
                prefill_observation={"seconds": 1},
            )
        ],
    )
    model = SimpleNamespace(
        program=copy.deepcopy(program),
        pending={},
        positions=np.array([0], dtype="int32"),
        occupied=np.array([True]),
        state=SimpleNamespace(generations=np.array([1])),
        batches=SimpleNamespace(
            states={"kv": SimpleNamespace(shape=(1, 2), dtype="float16")}
        ),
    )
    return model, directory, record


def test_import_accepts_decode_schedule_change(cache):
    model, directory, record = cache
    model.program["model_options"].update(
        decode_splits=512, cooperative_decode_attention=True
    )
    saved, first, provenance = load(model, directory, 2, 256, 0, record)
    assert saved.owner is model and first == [5]
    assert saved.positions.tolist() == [2]
    np.testing.assert_array_equal(saved.buffers["kv"], [[1, 2]])
    assert provenance["producer_run_id"] == "producer"


def test_import_rejects_different_precision(cache):
    model, directory, record = cache
    model.program["model_options"]["activation_quantization"] = "half"
    with pytest.raises(ValueError, match="numerical or storage"):
        load(model, directory, 2, 256, 0, record)


def test_import_rejects_modified_array(cache):
    model, directory, record = cache
    np.save(directory / "0.npy", np.zeros((1, 2), dtype="float16"))
    with pytest.raises(ValueError, match="path/hash"):
        load(model, directory, 2, 256, 0, record)


def test_import_rejects_unrelated_producer(cache):
    model, directory, record = cache
    record["results"][0]["prompt_sha256"] = "different"
    with pytest.raises(ValueError, match="sealed producer"):
        load(model, directory, 2, 256, 0, record)


def test_import_rejects_storage_shape_change(cache):
    model, directory, record = cache
    model.batches.states["kv"].shape = (2, 2)
    with pytest.raises(ValueError, match="shape/dtype"):
        load(model, directory, 2, 256, 0, record)
