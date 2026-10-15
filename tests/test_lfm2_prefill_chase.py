"""Experimental chunk profiles preserve decode and mixed-precision prefill."""

from akas.compilation.selection import source as kernel_source

import os
import numpy as np
import pytest
from support.lfm_profiles import valid_rows
from support.lfm_profiles import webgpu_parameters


@pytest.mark.parametrize("chunk", [32, 64, 128])
def test_experimental_chunk_is_explicit_and_decode_parameters_match(chunk):
    for profile in ("prefill_unrolled", "prefill_outer", "prefill_chunked"):
        assert valid_rows("webgpu", (1, chunk), profile)
        for kind, p in [
            ("linear", dict(r=1, k=1024, o=2560, type=1)),
            ("ffn", dict(r=1, k=1024, o=2560, type=1)),
            ("attention", dict(r=1, h=16, kh=8, d=64, cap=576)),
        ]:
            assert webgpu_parameters(kind, p, profile) == webgpu_parameters(
                kind, p, "decode_searched"
            )
    if chunk != 32:
        assert not valid_rows("webgpu", (1, chunk), "decode_searched")
    p = webgpu_parameters(
        "ffn", dict(r=chunk, k=1024, o=2560, type=1), "prefill_unrolled"
    )
    assert p["r"] == chunk and p["schedule"] == "partitioned"
    assert p["explicit_unroll"] and p["unroll"] == 16


def test_adaptive_rows_require_an_experimental_profile_and_sorted_unique_chunks():
    assert valid_rows("webgpu", (1, 32, 128), "prefill_outer")
    assert valid_rows("webgpu", (1, 32, 64, 128), "prefill_chunked")
    for rows in ((1, 128, 32), (1, 32, 32), (1, 33), (True, 32), (1,)):
        assert not valid_rows("webgpu", rows, "prefill_outer")
    assert not valid_rows("webgpu", (1, 32, 128), "decode_searched")
    assert not valid_rows("cuda", (1, 32, 128), "prefill_outer")


def test_selected_outer_configuration_is_not_shared_with_callers():
    p = dict(r=32, k=1024, o=1024, type=1)
    first = webgpu_parameters("linear", p, "prefill_outer")
    first["outer"]["tile_m"] = 1
    assert webgpu_parameters("linear", p, "prefill_outer")["outer"]["tile_m"] == 32
    ffn = webgpu_parameters("ffn", dict(r=128, k=1024, o=2560, type=1), "prefill_outer")
    assert ffn["schedule"] == "partitioned" and not ffn.get("explicit_unroll", False)


def test_quantized_search_selection_is_shape_specific_and_owned():
    assert valid_rows("webgpu", (1, 32, 128), "quant_searched")
    selected = webgpu_parameters(
        "linear", dict(r=1, k=2048, o=128000, type=14), "quant_searched"
    )
    assert selected["gemv_q6_dot"] and selected["gemv_unroll"] == 4
    selected = webgpu_parameters(
        "ffn", dict(r=128, k=2048, o=10752, type=2), "quant_searched"
    )
    selected["outer"]["tile_m"] = 1
    assert (
        webgpu_parameters(
            "ffn", dict(r=128, k=2048, o=10752, type=2), "quant_searched"
        )["outer"]["tile_m"]
        == 64
    )
    for kind, p in [
        ("linear", dict(r=1, k=1024, o=3072, type=2)),
        ("linear", dict(r=32, k=2048, o=2048, type=3)),
        ("linear", dict(r=64, k=2048, o=2048, type=2)),
        ("ffn", dict(r=1, k=1024, o=2560, type=1)),
        ("attention", dict(r=1, h=16, kh=8, d=64, cap=576)),
    ]:
        assert webgpu_parameters(kind, p, "quant_searched") == webgpu_parameters(
            kind, p, "prefill_chunked"
        )


def test_integer_and_mixed_prefill_are_opt_in_and_preserve_decode():
    for profile in ("prefill_q16", "prefill_mixed"):
        assert valid_rows("webgpu", (1, 32, 128), profile)
        for kind, p in [
            ("linear", dict(r=1, k=2048, o=6144, type=2)),
            ("linear", dict(r=1, k=2048, o=128000, type=14)),
            ("linear_add", dict(r=1, k=10752, o=2048, type=2)),
            ("ffn", dict(r=1, k=2048, o=10752, type=2)),
            ("attention", dict(r=1, h=32, kh=8, d=64, cap=640)),
        ]:
            assert webgpu_parameters(kind, p, profile) == webgpu_parameters(
                kind, p, "quant_searched"
            )
        p = dict(r=128, k=2048, o=512, type=2)
        selected = webgpu_parameters("linear", p, profile)
        assert selected["q16"]
        selected["integer"]["tile_m"] = 1
        assert webgpu_parameters("linear", p, profile)["integer"]["tile_m"] > 1
        assert webgpu_parameters(
            "linear", dict(p, type=14), profile
        ) == webgpu_parameters("linear", dict(p, type=14), "quant_searched")
    p = dict(r=128, k=2048, o=10752, type=2)
    selected = webgpu_parameters("ffn", p, "prefill_mixed")
    assert selected["outer"]["half_accum"] and not selected.get("q16")
    selected["outer"]["tile_m"] = 1
    assert webgpu_parameters("ffn", p, "prefill_mixed")["outer"]["tile_m"] > 1


@pytest.mark.parametrize("chunks,count,expected",[
    ((1,32),33,[32,1]),((1,32),17,[32]),
    ((1,128),129,[128,1]),((1,32,128),129,[128,1]),
    ((1,32,128),159,[128,32]),((1,32,128),384,[128,128,128]),
])
def test_common_prefill_qualified_chunks_and_final_readback(chunks,count,expected):
    from types import SimpleNamespace
    from akas.execution.executor import Executor

    calls=[]
    class Frame:
        def __init__(self,chunk,phase):
            self.chunk,self.phase=chunk,phase
            self.forward=SimpleNamespace(run=self.submit)
        def prepare(self,slots,sequences,positions):
            self.slots=tuple(slots)
            self.sequences=sequences
            self.start=int(positions[slots[0]])
        def submit(self):
            calls.append((self.chunk,self.start,len(self.sequences[0]),self.phase,False))
        def run(self,slots,sequences,positions):
            self.prepare(slots,sequences,positions)
            calls.append((self.chunk,self.start,len(sequences[0]),self.phase,True))
            return [4]*len(slots)
    records=[dict(phase="decode" if chunk==1 else "prefill",rows=1,chunk=chunk) for chunk in chunks]
    records += [dict(phase="prefill_state",rows=1,chunk=chunk) for chunk in chunks if chunk>1]
    frames={(r["phase"],r["chunk"]):Frame(r["chunk"],r["phase"]) for r in records}
    bank=SimpleNamespace(records=records,
                         select=lambda count,chunk,phase=None:frames[(phase or ("decode" if chunk==1 else "prefill"),chunk)])
    engine=Executor(capacity=1,context=512,batches=bank,reset=lambda slot:None)
    engine.config=SimpleNamespace(vocab=8)
    slot=engine.admit()
    assert engine.prefill([slot],[[4]*count],chunk=max(chunks)) == [4]
    assert [call[0] for call in calls] == expected
    assert [call[4] for call in calls] == [False]*(len(expected)-1)+[True]
    assert sum(call[2] for call in calls) == count
    assert [call[1] for call in calls] == [sum(v[2] for v in calls[:i]) for i in range(len(calls))]
    assert engine.positions[slot] == count
    assert engine.last_prefill_chunk == expected[-1]


@pytest.mark.skipif(
    os.environ.get("TENSOR_WEBGPU") != "1", reason="requires native WebGPU"
)
@pytest.mark.parametrize("kind", ["linear", "ffn"])
@pytest.mark.parametrize("layout,fma", [("km", True), ("mk", False)])
def test_mixed_activation_outer_product_fusion_and_tails(tmp_path, kind, layout, fma):
    import tensor
    from support.migration import source as source
    from support.projection_reference import reference

    r, k, o = 35, 129, 33
    p = dict(
        r=r,
        k=k,
        o=o,
        type=1,
        schedule="outer",
        outer=dict(
            tile_m=16,
            tile_n=32,
            tile_k=16,
            micro_m=2,
            micro_n=4,
            threads=64,
            lhs_layout=layout,
            fma=fma,
            unroll=8,
            explicit_unroll=True,
        ),
    )
    path = tmp_path / "outer.py"
    artifact = path.with_suffix(".tbin")
    path.write_text(
        kernel_source(
            kind,
            p,
            contract='packed',
            provider='webgpu',
        ).render()
    )
    tensor.build(path, artifact, provider="webgpu")
    rng = np.random.default_rng(903)
    weights = [
        rng.normal(size=(o, k)).astype(np.float16)
        for _ in range(2 if kind == "ffn" else 1)
    ]
    with tensor.Device(provider="webgpu") as device:
        kernel = device.load(artifact)
        ws = [device.from_numpy(w.ravel()) for w in weights]
        output = device.full(r * o, np.nan)
        for scale in (1, 0.01, 1e-5):
            x = (rng.normal(size=(r, k)) * scale).astype(np.float32)
            expected, bound = reference(x, weights)
            inp = device.from_numpy(x.ravel())
            device.write(output, np.full(r * o, np.nan, np.float32))
            kernel.launch(inp, *ws, output)
            actual = output.to_numpy().reshape(r, o)
            assert np.isfinite(actual).all()
            assert np.all(np.abs(actual - expected) <= bound)
            inp.release()


@pytest.mark.skipif(
    os.environ.get("TENSOR_WEBGPU") != "1", reason="requires native WebGPU"
)
@pytest.mark.parametrize("kind", ["linear", "ffn"])
@pytest.mark.parametrize("unroll", [2, 4, 8])
@pytest.mark.parametrize("magnitude", [0.01, 1.0, 1e-5])
@pytest.mark.parametrize("group_order", ["column", "row"])
def test_short_half_fma_chains_against_explicit_rounding(
    tmp_path, kind, unroll, magnitude, group_order
):
    import tensor
    from support.migration import source as source

    r, k, o = 35, 37, 34
    rng = np.random.default_rng(1683)
    weights = [
        (rng.normal(size=(o, k)) * 0.02).astype(np.float16)
        for _ in range(2 if kind == "ffn" else 1)
    ]
    x = (rng.normal(size=(r, k)) * magnitude).astype(np.float32)
    aa = x.astype(np.float16).astype(np.float64)
    values = []
    for w in weights:
        bb = w.astype(np.float64)
        total = np.zeros((r, o), np.float32)
        for start in range(0, k, 2 * unroll):
            partial = np.zeros((r, o, 2), np.float16)
            for step in range(unroll):
                for lane in range(2):
                    index = start + step * 2 + lane
                    if index < k:
                        partial[:, :, lane] = (
                            aa[:, index, None] * bb[None, :, index]
                            + partial[:, :, lane].astype(np.float64)
                        ).astype(np.float16)
            total += partial[:, :, 0].astype(np.float32) + partial[:, :, 1].astype(
                np.float32
            )
        values.append(total.astype(np.float64))
    expected = (
        values[0]
        if kind == "linear"
        else values[0] / (1 + np.exp(-values[0])) * values[1]
    )
    p = dict(
        r=r,
        k=k,
        o=o,
        type=1,
        schedule="outer",
        outer=dict(
            tile_m=16,
            tile_n=32,
            tile_k=16,
            micro_m=2,
            micro_n=4,
            threads=64,
            unroll=unroll,
            explicit_unroll=True,
            dot_width=2,
            packed_pairs=True,
            half_accum=True,
            group_order=group_order,
            lhs_pad=int(group_order == "row"),
            rhs_pad=int(group_order == "row"),
        ),
    )
    path = tmp_path / "half.py"
    path.write_text(
        kernel_source(
            kind,
            p,
            contract='packed',
            provider='webgpu',
        ).render()
    )
    artifact = path.with_suffix(".tbin")
    tensor.build(path, artifact, provider="webgpu")
    with tensor.Device(provider="webgpu") as device:
        kernel = device.load(artifact)
        inp = device.from_numpy(x.ravel())
        ws = [device.from_numpy(w.ravel()) for w in weights]
        output = device.full(r * o, np.nan)
        kernel.launch(inp, *ws, output)
        actual = output.to_numpy().reshape(r, o)
        assert np.isfinite(actual).all()
        np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=1e-9)


@pytest.mark.skipif(
    os.environ.get("TENSOR_WEBGPU") != "1", reason="requires native WebGPU"
)
def test_parallel_prefill_rms_matches_independent_reduction(tmp_path):
    import tensor
    from support.migration import source as source

    r, c = 35, 2048
    rng = np.random.default_rng(1725)
    w = rng.normal(size=c).astype(np.float32)
    path = tmp_path / "rms.py"
    path.write_text(
        kernel_source(
            "rms",
            dict(r=r, c=c, eps=1e-5, sg=True, parallel_rows=True),
            contract='packed',
            provider='webgpu',
        ).render()
    )
    artifact = path.with_suffix(".tbin")
    tensor.build(path, artifact, provider="webgpu")
    with tensor.Device(provider="webgpu") as device:
        kernel = device.load(artifact)
        weights = device.from_numpy(w)
        out = device.full(r * c, np.nan)
        for scale in (0.01, 1, 1e-5):
            x = (rng.normal(size=(r, c)) * scale).astype(np.float32)
            x[0] = 0
            expected = (
                x.astype(np.float64)
                * w
                / (np.mean(x.astype(np.float64) ** 2, axis=1)[:, None] + 1e-5) ** 0.5
            )
            inp = device.from_numpy(x.ravel())
            kernel.launch(inp, weights, out)
            np.testing.assert_allclose(
                out.to_numpy().reshape(r, c), expected, rtol=3e-6, atol=1e-10
            )
            inp.release()


@pytest.mark.skipif(
    os.environ.get("TENSOR_WEBGPU") != "1", reason="requires native WebGPU"
)
@pytest.mark.parametrize("encoding", [2, 14])
@pytest.mark.parametrize("kind", ["linear", "ffn"])
@pytest.mark.parametrize(
    "shared_dtype,dot_width,packed_pairs",
    [
        ("float16", 1, False),
        ("float16", 2, False),
        ("float16", 4, False),
        ("float32", 1, False),
        ("float32", 2, False),
        ("float32", 4, False),
        ("float16", 2, True),
    ],
)
def test_packed_outer_product_rounding_fusion_and_output_tail(
    tmp_path, encoding, kind, shared_dtype, dot_width, packed_pairs
):
    import tensor
    from akas.checkpoints.gguf import TYPES
    from akas.checkpoints.gguf import dequantize
    from support.migration import source as source
    from support.projection_reference import reference

    r, k, o = 35, 288 if encoding == 2 else 256, 34
    _, block, size = TYPES[encoding]
    rng = np.random.default_rng(914)
    raw = []
    weights = []
    for _ in range(2 if kind == "ffn" else 1):
        packed = rng.integers(0, 256, (o * k // block, size), dtype=np.uint8)
        scales = np.resize(
            np.array([0.0, 1.0, -1.0, 2**-24, -(2**-24), 0.001], np.float16),
            len(packed),
        )
        offset = 0 if encoding == 2 else 208
        packed[:, offset : offset + 2] = scales.view(np.uint8).reshape(-1, 2)
        raw.append(packed.ravel().view(np.uint32))
        weights.append(
            dequantize(packed.ravel(), encoding).reshape(o, k).astype(np.float16)
        )
    p = dict(
        r=r,
        k=k,
        o=o,
        type=encoding,
        schedule="outer",
        outer_shared_dtype=shared_dtype,
        outer=dict(
            tile_m=16,
            tile_n=32,
            tile_k=64,
            micro_m=2,
            micro_n=4,
            threads=64,
            lhs_layout="km",
            fma=True,
            unroll=8,
            explicit_unroll=True,
            dot_width=dot_width,
            packed_pairs=packed_pairs,
        ),
    )
    path = tmp_path / "packed.py"
    artifact = path.with_suffix(".tbin")
    path.write_text(
        kernel_source(
            kind,
            p,
            contract='packed',
            provider='webgpu',
        ).render()
    )
    tensor.build(path, artifact, provider="webgpu")
    with tensor.Device(provider="webgpu") as device:
        kernel = device.load(artifact)
        uploaded = [device.from_numpy(w) for w in raw]
        output = device.full(r * o, np.nan)
        for scale in (0.01, 1e-5):
            x = (rng.normal(size=(r, k)) * scale).astype(np.float32)
            expected, bound = reference(x, weights)
            inp = device.from_numpy(x.ravel())
            kernel.launch(inp, *uploaded, output)
            actual = output.to_numpy().reshape(r, o)
            assert np.isfinite(actual).all() and np.all(
                np.abs(actual - expected) <= bound
            )
            inp.release()
