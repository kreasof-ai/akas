"""A three-file model: y = x + x @ projection.T, using existing operations."""

from akas.planning import ForwardPlan


def compose(config, *, rows=1):
    plan = ForwardPlan(contract="packed")
    x = plan.buffer("input", (rows * config.width,), "float32", role="input")
    w = plan.buffer(
        "projection", (config.width * config.width,), "float32", role="weight"
    )
    hidden = plan.buffer("projected", x.shape, "float32")
    out = plan.buffer("output", x.shape, "float32", role="output")
    plan("linear", dict(r=rows, k=config.width, o=config.width, type=0), x, w, hidden)
    plan("add", dict(r=rows, c=config.width), x, hidden, out)
    return plan
