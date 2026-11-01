"""Typed producer outputs preserve an existing consumer's rounding boundary."""
import os
import pytest


def normalization_graph(*, named=False, guard=None):
    from akas.graph import Graph

    graph = Graph(contract="reference")
    x = graph.buffer("x", (4 * 513,), "float32", role="input")
    weight = graph.buffer("weight", (513,), "float32", role="weight")
    normalized = graph.buffer("normalized", x.shape, "float32",
                              role="output" if guard == "public" else "scratch")
    converted = graph.buffer("converted", x.shape, "float16", role="output")
    p = dict(r=4, c=513, eps=1e-5, algorithm="serial-tree", stage_values=True)
    if named:
        graph("rms", p, x=x, w=weight, out=normalized)
        graph("linear_half_input", dict(r=4, k=513), x=normalized, out=converted)
    else:
        graph("rms", p, x, weight, normalized)
        graph("linear_half_input", dict(r=4, k=513), normalized, converted)
    if guard == "feature":
        alias = normalized.view((4, 513))
        graph.bind_requests(inputs={}, prediction="converted", features={"visible": alias.name})
    elif guard == "other_user":
        graph("unknown_consumer", {}, normalized, x)
    return graph


@pytest.mark.parametrize("named", [False, True])
@pytest.mark.parametrize("guard", [None, "public", "feature", "other_user"])
def test_staging_fusion_keeps_public_values_aliases_and_other_consumers(named, guard):
    from akas.compilation.policy import lower
    from akas.compilation.compile import plan

    graph = normalization_graph(named=named, guard=guard)
    lower(graph, policy={"normalization": dict(fused_staging=True)})
    if guard:
        assert graph.calls[1].kind == "linear_half_input"
        assert graph.buffers["normalized"].dtype == "float32"
    else:
        assert len(graph.calls) == 1
        assert graph.calls[0].parameters["output_dtype"] == "float16"
        assert (graph.calls[0].bindings.get("out") if named else graph.calls[0].arguments[-1]) == "converted"
        plan(graph, target="sm_89")


@pytest.mark.skipif(os.getenv("AKAS_LFM_FP8_CUDA") != "1", reason="CUDA storage fusion qualification")
@pytest.mark.parametrize("kind", ["rms", "conv_decode", "conv_prefill"])
def test_output_staging_fusion_is_bitwise_identical(tmp_path, kind):
    from contextlib import ExitStack
    import numpy as np
    import tensor
    from akas.graph import Graph
    from akas.compilation.policy import lower
    from akas.compilation.compile import produce
    from akas.execution.forward import Forward

    rng = np.random.default_rng(911)
    if kind == "rms":
        graph = normalization_graph()
        host = dict(x=rng.normal(0, .5, (4 * 513,)).astype("float32"),
                    weight=rng.uniform(.1, 1.5, (513,)).astype("float32"))
        policy = {"normalization": dict(fused_staging=True)}
    else:
        rows, width, chunk = 3, 129, 65 if kind == "conv_prefill" else 1
        graph = Graph(contract="reference")
        host = dict(x=rng.normal(0, .5, (rows * chunk * 3 * width,)).astype("float32"),
                    weight=rng.normal(0, .2, (width * 3,)).astype("float32"),
                    state=rng.normal(0, .5, (rows, 2, width)).astype("float32"),
                    enabled=np.array([chunk, 0, max(0, chunk - 1)], dtype="int32"))
        for name, array in host.items():
            graph.buffer(name, array.shape, str(array.dtype), role="state" if name == "state" else "input")
        mixed = graph.buffer("mixed", (rows * chunk * width,), "float32")
        converted = graph.buffer("converted", mixed.shape, "float16", role="output")
        p = dict(r=rows, c=width, chunk=chunk, length=3)
        if chunk > 1:
            p.update(time_tile=32, columns=16, threads=128)
        graph("conv", p, graph.buffers["x"], graph.buffers["weight"], graph.buffers["state"], mixed, graph.buffers["enabled"])
        graph("linear_half_input", dict(r=rows * chunk, k=width), mixed, converted)
        policy = {"convolution": dict(fused_staging=True)}
    fused, _ = lower(Graph.from_metadata(graph.metadata()), policy=policy)
    assert len(fused.calls) == len(graph.calls) - 1
    outputs = []
    with tensor.Device() as device:
        for name, candidate in (("separate", graph), ("fused", fused)):
            with ExitStack() as lifetime:
                buffers = {key: device.from_numpy(value) for key, value in host.items()}
                buffers["converted"] = device.zeros(graph.buffers["converted"].shape, "float16")
                for value in buffers.values():
                    lifetime.callback(value.release)
                destination = tmp_path / name
                produce(candidate, destination, search=False, target="sm_89")
                with Forward(destination, device, buffers, graphs=True) as forward:
                    forward.run()
                    outputs.append({key: buffers[key].to_numpy() for key in ("converted", "state") if key in buffers})
        for key in outputs[0]:
            np.testing.assert_array_equal(outputs[0][key], outputs[1][key])



@pytest.mark.skipif(os.getenv('AKAS_LFM_FP8_CUDA') != '1', reason='CUDA storage fusion model qualification')
def test_packed_and_staging_fusion_model_is_bitwise_for_ragged_state_and_reuse(tmp_path):
    import shutil
    import numpy as np
    import tensor
    from akas.models.lfm2 import LFM2MoE
    from akas.compilation import compile_checkpoint
    from test_lfm_moe_native import fixture

    path=fixture(tmp_path/'weights');bundle=tmp_path/'program';fused=tmp_path/'fused-program'
    policy=dict(moe=dict(fused_swiglu=True,activation_dtype='bfloat16',intermediate_dtype='bfloat16',packed=True,packed_minimum_rows=1),
                linear=dict(stage_inputs=True,floating_matmul=True),
                normalization=dict(fused_residual=True,fused_staging=True),convolution=dict(fused_staging=True))
    options=dict(checkpoint_options={'tokenizer':False},geometries=[(1,1),(2,1),(2,64)],
                 max_requests=4,max_context=64,target='sm_89',search=False)
    compile_checkpoint(LFM2MoE,path,bundle,policy=policy,**options)
    shutil.copytree(bundle,fused)
    baseline={**policy,'moe':{k:v for k,v in policy['moe'].items() if k not in ('packed','packed_minimum_rows')},'normalization':dict(fused_residual=True)};baseline.pop('convolution')
    compile_checkpoint(LFM2MoE,path,bundle,policy=baseline,**options)
    outputs=[]
    for program in (fused,bundle):
        with tensor.Device() as device,LFM2MoE(path,program,device,checkpoint_options={'tokenizer':False}) as model:
            frames=[];slots=[model.admit_at(3),model.admit_at(1)]
            ids=model.prefill(slots,[[1,3,5],[2,4]],chunk=64);frames.append(model.last_frame.feature('logits').to_numpy())
            model.decode(slots[::-1],ids[::-1]);frames.append(model.last_frame.feature('logits').to_numpy())
            model.release(3);assert model.admit_at(3)==3
            model.prefill([3],[[7,9,11]],chunk=64);frames.append(model.last_frame.feature('logits').to_numpy())
            outputs.append(frames)
    for expected,actual in zip(outputs[0],outputs[1]):np.testing.assert_array_equal(actual,expected)
