"""LFM verification exposes token predictions and leaves histories to commit."""

import pytest
from akas.compilation.compile import plan
from akas.models.lfm2.model import LFM2Reference
from akas.models.lfm2.weights import ReferenceCheckpoint
from support.diagnostic import fixture


@pytest.mark.parametrize("provider", ["cuda", "webgpu"])
@pytest.mark.parametrize("chunk", [1, 3])
def test_lfm_verifier_uses_private_journals_and_preserves_target_features(
    tmp_path, provider, chunk
):
    checkpoint = ReferenceCheckpoint(fixture(tmp_path / "model.gguf"), tokenizer=False)
    composed = LFM2Reference.compose(
        checkpoint,
        rows=2,
        chunk=chunk,
        max_requests=4,
        max_context=32,
        provider=provider,
        phase="verify",
        auxiliary_layers=(0, 1),
    )
    graph, commit = composed.graph, composed.commit
    kinds = [call.kind for call in graph.calls]
    assert "conv_verify" in kinds and "conv" not in kinds
    assert not {"advance", "next_tokens", "chunk_last"} & set(kinds)
    assert graph.buffers["sampled"].shape == (2 * chunk,)
    assert graph.buffers["logits"].shape == (2 * chunk * checkpoint.config.vocab,)
    assert graph.buffers["blk.0.conv_state"].shape == (4, 2, checkpoint.config.width)
    assert graph.buffers["blk.0.conv_journal"].shape == (
        2,
        chunk,
        2,
        checkpoint.config.width,
    )
    assert graph.buffers["blk.0.conv_journal"].role == "journal"
    for layer in (0, 1):
        feature = graph.request["features"][f"layer.{layer}"]
        assert graph.buffers[feature].role == "output"
    assert [c.kind for c in commit.calls] == ["conv_commit", "advance"]
    assert commit.request is None
    assert commit.calls[0].bindings["accepted"] == "active"
    # Request mapping must lift only persistent state into the slot pool.
    for forward in (graph, commit):
        bundle = plan(forward, provider=provider)
        assert len(bundle.kernels) == len(forward.requirements())


def test_lfm_rejects_invalid_target_capture_indices(tmp_path):
    checkpoint = ReferenceCheckpoint(fixture(tmp_path / "model.gguf"), tokenizer=False)
    for indices in ((-1,), (2,), (0, 0)):
        with pytest.raises(ValueError, match="auxiliary layers"):
            LFM2Reference.compose(
                checkpoint,
                rows=1,
                chunk=1,
                max_requests=1,
                max_context=32,
                auxiliary_layers=indices,
            )
