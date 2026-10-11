"""Shared operations preserve numerical profiles, ABI order and scratch validity."""

from types import SimpleNamespace

from akas.artifacts.kernels import identity
from akas.execution.plan import KernelPlan
from akas.ops.linear import linear, split_linear
from akas.ops.normalization import rms_norm
from akas.ops.quantization import ActivationCache


class Plan:
    def __init__(self):
        self.calls = []

    def bind(self, kernel, arguments, *, label=None):
        self.calls.append((kernel, arguments, label))


def kernel(*names):
    return SimpleNamespace(manifest={"arguments": [{"name": name} for name in names]})


def test_fp8_decode_and_prefill_keep_distinct_activation_precision():
    weight = SimpleNamespace(shape=(64, 128))
    source, destination, scales, activation_scales = [object() for _ in range(4)]
    parameters = dict(r=4, k=128, o=64)
    for activation, names, expected in [
        (None, ("w", "x", "scales", "out"), (weight, source, scales, destination)),
        (
            activation_scales,
            ("x", "w", "activation_scales", "scales", "out"),
            (source, weight, activation_scales, scales, destination),
        ),
    ]:
        selected = kernel(*names)
        plan = Plan()
        emit = KernelPlan(plan, {identity("fp8_linear", parameters): selected})
        linear(
            emit,
            source,
            weight,
            destination,
            rows=4,
            scales=scales,
            activation_scales=activation,
        )
        assert plan.calls == [(selected, expected, None)]


def test_residual_normalization_preserves_label_and_profile_parameters():
    x, residual, weight, out = [object() for _ in range(4)]
    parameters = dict(r=2, c=128, eps=1e-6)
    selected = kernel("residual", "w", "x", "out")
    plan = Plan()
    seen = []

    def profile(kind, values):
        seen.append((kind, values.copy()))
        return dict(values, provider="test")

    emit = KernelPlan(
        plan,
        {identity("add_rms", dict(parameters, provider="test")): selected},
        parameters=profile,
    )
    rms_norm(
        emit,
        x,
        weight,
        out,
        rows=2,
        width=128,
        epsilon=1e-6,
        residual=residual,
        label=(0, "ffn"),
    )
    assert seen == [("add_rms", parameters)]
    assert plan.calls == [(selected, (residual, weight, x, out), (0, "ffn"))]


def test_quantized_scratch_reuses_reads_and_invalidates_all_overwritten_aliases():
    source = SimpleNamespace(shape=(4, 8, 128))
    unused = SimpleNamespace(shape=(4, 128))
    sources = dict(first=source, alias=source, other=unused)
    outputs = {name: (object(), object()) for name in sources}
    cache = ActivationCache(sources, outputs, rows=4)
    selected = kernel("x", "out", "scales")
    parameters = dict(r=4, k=128, top=8)
    plan = Plan()
    emit = KernelPlan(
        plan, {identity("quantize", parameters): selected}, on_write=cache.invalidate
    )
    cache.get("first", emit)
    cache.get("first", emit)
    cache.get("alias", emit)
    assert len(plan.calls) == 2 and cache.valid == {"first", "alias"}
    cache.invalidate(unused)
    assert cache.valid == {"first", "alias"}
    cache.invalidate(source)
    assert not cache.valid
    cache.get("first", emit)
    assert len(plan.calls) == 3
    assert plan.calls[-1][1] == (source, *outputs["first"])


def test_single_split_borrows_output_and_multi_split_emits_merge(monkeypatch):
    import tensor.runtime

    monkeypatch.setattr(
        tensor.runtime,
        "Buffer",
        lambda device, pointer, shape, dtype, owner: SimpleNamespace(
            pointer=pointer, shape=shape, dtype=dtype, owner=owner
        ),
    )
    calls = []

    def emit(kind, parameters, *args):
        calls.append((kind, parameters, args))

    source = object()
    weight = SimpleNamespace(shape=(64, 128))
    output = SimpleNamespace(pointer=123, dtype="float32")
    scratch = object()
    views = []
    split_linear(
        emit,
        object(),
        source,
        weight,
        output,
        rows=4,
        parameters=dict(parts=1),
        scratch=None,
        views=views,
    )
    assert len(calls) == 1 and calls[0][0] == "split_linear"
    assert calls[0][2][-1] is views[0] and views[0].owner is output
    assert views[0].shape == (4, 1, 64)
    calls.clear()
    split_linear(
        emit,
        object(),
        source,
        weight,
        output,
        rows=4,
        parameters=dict(parts=2),
        scratch=scratch,
        views=views,
    )
    assert [entry[0] for entry in calls] == ["split_linear", "split_merge"]
    assert calls[1][2] == (scratch, output) and len(views) == 1
