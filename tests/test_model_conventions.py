"""New model packages use the common composition and request lifecycle."""

import json
import os
import struct

import numpy as np
import pytest

from akas.models.definition import Composition
from examples.models.bigram import Bigram


def checkpoint(directory):
    vocab = 8
    (directory / "config.json").write_text(json.dumps(dict(vocab=vocab, context=32)))
    scores = np.full((vocab, vocab), -1, dtype="float32")
    scores[np.arange(vocab), (np.arange(vocab) + 1) % vocab] = 2
    header = json.dumps(
        {
            "scores": dict(
                dtype="F32", shape=list(scores.shape), data_offsets=[0, scores.nbytes]
            )
        }
    ).encode()
    (directory / "model.safetensors").write_bytes(
        struct.pack("<Q", len(header)) + header + scores.tobytes()
    )
    return Bigram.checkpoint_type(directory)


def test_new_model_uses_one_composition_entry_and_result(tmp_path):
    loaded = checkpoint(tmp_path)
    composition = Bigram.compose(
        loaded, rows=4, chunk=1, max_requests=4, max_context=32
    )
    assert isinstance(composition, Composition)
    assert composition.commit is None
    assert composition.graph.request["prediction"] == "predicted"
    assert not hasattr(Bigram, "definition")
    with pytest.raises(ValueError, match="one token"):
        Bigram.compose(loaded, rows=4, chunk=4, max_requests=4, max_context=32)


@pytest.mark.skipif(
    os.environ.get("AKAS_FORWARD_CUDA") != "1",
    reason="example request model GPU qualification",
)
def test_new_model_runs_reordered_requests_through_shared_executor(tmp_path):
    import tensor
    from akas.compilation import compile_checkpoint

    checkpoint(tmp_path)
    out = tmp_path / "compiled"
    compile_checkpoint(
        Bigram,
        tmp_path,
        out,
        geometries=[(4, 1)],
        max_requests=4,
        max_context=32,
        search=False,
    )
    with tensor.Device() as device, Bigram(tmp_path, out, device) as model:
        slots = [model.admit_at(3), model.admit_at(1)]
        assert model.prefill(slots, [[1, 2, 3], [5]], chunk=1) == [4, 6]
        assert model.decode(slots[::-1], [6, 4]) == [7, 5]
        assert model.positions.tolist() == [0, 2, 0, 4]
