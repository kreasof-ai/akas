"""A complete request model using two existing operations and the common executor."""

from akas.graph import Graph
from akas.models.definition import Composition, Definition
from examples.models.bigram.weights import Checkpoint


def compose(
    checkpoint, *, rows, chunk, max_requests, max_context, provider="cuda", phase=None
):
    if chunk != 1 or phase not in (None, "decode"):
        raise ValueError("the bigram example supports one token per request step")
    if max_context > checkpoint.config.context:
        raise ValueError("requested context exceeds the checkpoint configuration")

    vocab = checkpoint.config.vocab
    graph = Graph(contract="reference", phase="decode")
    scores = graph.buffer("scores", (vocab * vocab,), "float32", role="weight")
    tokens = graph.buffer("tokens", (rows,), "int32", role="input")
    graph.buffer("mapping", (rows,), "int32", role="input")
    lengths = graph.buffer("lengths", (rows,), "int32", role="input")
    logits = graph.buffer("logits", (rows * vocab,), "float32")
    predicted = graph.buffer("predicted", (rows,), "int32", role="output")

    # Looking up a row gives the next-token scores for the current token.
    graph("embedding", dict(r=rows, c=vocab, v=vocab), tokens, scores, logits)
    graph("argmax", dict(r=rows, c=vocab), logits, predicted, lengths)
    graph.bind_requests(
        inputs={"tokens": "tokens", "mapping": "mapping", "lengths": "lengths"},
        prediction="predicted",
        features={"logits": "logits"},
    )
    return Composition(graph)


class Bigram(Definition):
    checkpoint_type = Checkpoint
    optimization_chunks = (1,)
