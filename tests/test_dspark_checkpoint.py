"""Draft metadata, exact target feature order and borrowed weight lifetimes."""

from types import SimpleNamespace
import pytest
from akas.checkpoints.gguf import GGUFError
from akas.graph import Graph
from akas.models.dspark.config import DSparkDraftConfig
from akas.models.dspark.weights import DSparkDraftCheckpoint, tensor_shapes


@pytest.fixture
def paired(monkeypatch):
    import akas.models.dspark.weights as weights

    target_config = SimpleNamespace(
        width=64, vocab=16, layers=("conv", "attention"), max_context=1024
    )
    metadata = {
        "general.architecture": "dflash",
        "dflash.sample_from_anchor": True,
        "dflash.target_layers": [1],
        "dflash.embedding_length": 64,
        "dflash.feed_forward_length": 32,
        "dflash.attention.head_count": 1,
        "dflash.attention.head_count_kv": 1,
        "dflash.block_count": 1,
        "dflash.context_length": 2048,
        "dflash.block_size": 3,
        "dflash.attention.layer_norm_rms_epsilon": 1e-5,
        "dflash.rope.freq_base": 10000.0,
        "tokenizer.ggml.mask_token_id": 7,
        "tokenizer.ggml.tokens": list(map(str, range(16))),
    }
    gguf = SimpleNamespace(
        metadata=metadata, tensors={"markov_w1.weight": SimpleNamespace(shape=(16, 4))}
    )
    config = DSparkDraftConfig.from_gguf(gguf, target_config)
    gguf.tensors = {
        n: SimpleNamespace(shape=s, type=0 if len(s) == 1 else 1)
        for n, s in tensor_shapes(config).items()
    }
    target = SimpleNamespace(
        config=target_config,
        gguf=SimpleNamespace(
            tensors={"token_embd.weight": SimpleNamespace(shape=(16, 64), type=1)}
        ),
        tokenizer=None,
    )
    monkeypatch.setattr(weights, "GGUF", lambda _: gguf)
    return target, gguf


def test_dspark_reader_preserves_metadata_feature_order_and_target_limits(paired):
    target, gguf = paired
    gguf.metadata["dflash.target_layers"] = [2, 1]
    config = DSparkDraftConfig.from_gguf(gguf, target.config)
    assert config.target_layers == (1, 0)
    assert config.feature_width == 128 and config.max_context == 1024
    assert config.mask_token == 7 and config.markov_rank == 4


@pytest.mark.parametrize(
    "field,value",
    [
        ("dflash.target_layers", [0]),
        ("dflash.target_layers", [3]),
        ("dflash.target_layers", [1, 1]),
        ("dflash.sample_from_anchor", False),
        ("dflash.embedding_length", 128),
        ("dflash.attention.head_count", 0),
        ("tokenizer.ggml.mask_token_id", 16),
        ("dflash.rope.freq_base", float("nan")),
    ],
)
def test_dspark_reader_rejects_incompatible_sidecar_contract(paired, field, value):
    target, gguf = paired
    gguf.metadata[field] = value
    with pytest.raises(GGUFError):
        DSparkDraftCheckpoint("sidecar.gguf", target=target)


def test_dspark_reader_keeps_shared_tied_head_as_one_borrowed_buffer(paired):
    target, _ = paired
    checkpoint = DSparkDraftCheckpoint("sidecar.gguf", target=target)
    weights = checkpoint.graph_weights(Graph(contract="reference"), provider="cuda")
    assert weights["embedding"] is weights["head"]
    assert len([n for n in weights.buffers if n.startswith("target.")]) == 1
    assert weights["fc.weight"].dtype == "float16"
    assert weights["fc.weight"].shape == (64 * 64,)
    assert weights.parameters("head") == dict(ggml=True, type=1)
    with pytest.raises(ValueError, match="loaded target executor"):
        checkpoint.upload(object())


def test_dspark_reader_rejects_missing_or_quantized_draft_norm(paired):
    target, gguf = paired
    gguf.tensors["output_norm.weight"].type = 1
    with pytest.raises(GGUFError, match="native floating"):
        DSparkDraftCheckpoint("sidecar.gguf", target=target)
    del gguf.tensors["output_norm.weight"]
    with pytest.raises(GGUFError, match="missing"):
        DSparkDraftCheckpoint("sidecar.gguf", target=target)


class Buffer:
    def __init__(self, device, *, fail=False):
        self.device = device
        self.shape = (1024,)
        self.released = False
        self.children = []
        self.fail = fail

    def view(self, shape):
        if self.fail:
            raise RuntimeError("view failure")
        value = Buffer(self.device)
        self.children.append(value)
        return value

    def release(self):
        self.released = True


def test_dspark_upload_releases_borrowed_views_without_freeing_target(
    paired, monkeypatch
):
    import akas.models.dspark.weights as module

    target, gguf = paired
    device = object()
    base = Buffer(device)
    executor = SimpleNamespace(checkpoint=target, weights={"token_embd.weight": base})
    own = {"draft-weight": Buffer(device)}
    calls = []

    def upload(actual_device, actual_checkpoint):
        calls.append((actual_device, actual_checkpoint))
        return own

    monkeypatch.setattr(module, "upload_packed", upload)
    checkpoint = DSparkDraftCheckpoint("sidecar.gguf", target=executor)
    result = checkpoint.upload(device)
    assert calls == [(device, gguf)] and len(base.children) == 1
    assert result["target.token_embd.weight"] is base.children[0]
    for buffer in result.values():
        buffer.release()
    assert (
        own["draft-weight"].released and base.children[0].released and not base.released
    )


def test_dspark_upload_failure_releases_new_buffers_and_prior_views(
    paired, monkeypatch
):
    import akas.models.dspark.weights as module

    target, _ = paired
    device = object()
    target.gguf.tensors["output.weight"] = SimpleNamespace(shape=(16, 64), type=1)
    head, embedding = Buffer(device), Buffer(device, fail=True)
    executor = SimpleNamespace(
        checkpoint=target,
        weights={"token_embd.weight": embedding, "output.weight": head},
    )
    own = Buffer(device)
    monkeypatch.setattr(module, "upload_packed", lambda *_: {"weight": own})
    checkpoint = DSparkDraftCheckpoint("sidecar.gguf", target=executor)
    with pytest.raises(RuntimeError, match="view failure"):
        checkpoint.upload(device)
    assert own.released and head.children[0].released
    assert not head.released and not embedding.released
