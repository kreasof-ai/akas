"""Shared aligned expert packing preserves bytes, scales and routing."""
import os
import pytest
from test_lfm_moe_native import test_native_operations_match_independent_references as check_operation

@pytest.mark.skipif(os.getenv('AKAS_LFM_FP8_CUDA') != '1', reason='CUDA routed packing qualification')
@pytest.mark.parametrize('rows,maximum,routed', [(1,16,False),(33,64,False),(129,128,True)])
def test_expert_pack_preserves_quantized_values_and_scales(tmp_path, rows, maximum, routed):
    check_operation(tmp_path, 'expert_pack', dict(r=rows,k=256,top=3,experts=7,max_block_m=maximum,routed_input=routed))


@pytest.mark.skipif(os.getenv('AKAS_LFM_FP8_CUDA') != '1', reason='CUDA routed packing qualification')
@pytest.mark.parametrize('kind,routed,packed_output', [('expert_partial',True,False),('expert_partial',False,False),('fp8_expert_swiglu',False,True),('expert_swiglu',False,False)])
def test_packed_projection_matches_independent_equations(tmp_path,kind,routed,packed_output):
    p=dict(r=33,rows=33,k=384,o=256,experts=7,top=3,partitions=1,block_m=32,max_block_m=64,
           compact=True,tile_map=True,columns=128,threads=128,stages=1,activation_vector=16,
           packed_input=True,packed_output=packed_output,routed_input=routed,
           product_dtype='bfloat16',activation_dtype='bfloat16')
    if kind=='expert_swiglu':p.update(weight_format='fp8',input_dtype='uint8',output_dtype='bfloat16')
    check_operation(tmp_path,kind,p)


@pytest.mark.skipif(os.getenv('AKAS_LFM_FP8_CUDA') != '1', reason='CUDA routed packing qualification')
def test_packed_expert_chain_matches_original_bits(tmp_path):
    from contextlib import ExitStack
    import numpy as np
    import tensor
    from tensor.runtime.dtypes import encode_bfloat16, decode_bfloat16
    from akas.graph import Graph
    from akas.compilation.policy import lower
    from akas.compilation.compile import produce
    from akas.execution.forward import Forward
    from akas.ops.moe.validation import tile_tables

    rows, top, experts, depth, hidden, width = 33, 3, 7, 384, 256, 256
    rng = np.random.default_rng(1035)
    def bits(shape):
        return rng.integers(32,81,shape,dtype='uint8') | rng.integers(0,2,shape,dtype='uint8') * 128
    def scales(shape):
        return decode_bfloat16(encode_bfloat16(rng.uniform(.0625,.25,shape).astype('float32')))
    counts=np.zeros(experts,'int32');routes=np.full((experts,rows),-1,'int32')
    for row in range(rows-1):
        for rank in range(top):
            expert=(row*3+rank)%experts;routes[expert,counts[expert]]=row*top+rank;counts[expert]+=1
    tables=tile_tables(counts,(rows*top+3)//4+experts)
    host=dict(x=bits((rows,depth)),activation_scales=scales((rows,depth//128)),
              w=bits((experts,hidden,depth)),scales=scales((experts,hidden//128,depth//128)),
              w2=bits((experts,hidden,depth)),scales2=scales((experts,hidden//128,depth//128)),
              down=bits((experts,width,hidden)),down_scales=scales((experts,width//128,hidden//128)),
              counts=counts,routes=routes,tile_experts=tables[0],tile_offsets=tables[1])
    graph=Graph(contract='reference')
    for name,array in host.items():graph.buffer(name,array.shape,'bfloat16' if name in ('scales','scales2','down_scales') else str(array.dtype),role='input')
    activated=graph.buffer('activated',(rows,top,hidden),'uint8')
    activation_scales=graph.buffer('activated_scales',(rows,top,hidden//128),'float32')
    out=graph.buffer('out',(rows,top,width),'bfloat16',role='output')
    common=dict(r=rows,rows=rows,top=top,experts=experts,partitions=1,block_m=32,
                compact=True,tile_map=True,columns=128,threads=128,stages=1,activation_vector=16)
    shared={name:graph.buffers[name] for name in ('counts','routes','tile_experts','tile_offsets')}
    graph('fp8_expert_swiglu',dict(common,k=depth,o=hidden,product_dtype='bfloat16',activation_dtype='bfloat16'),
          **{name:graph.buffers[name] for name in ('x','activation_scales','w','scales','w2','scales2')},
          **shared,out=activated,output_scales=activation_scales)
    graph('expert_partial',dict(common,k=hidden,o=width,routed_input=True,output_dtype='bfloat16'),
          x=activated,activation_scales=activation_scales,w=graph.buffers['down'],scales=graph.buffers['down_scales'],**shared,out=out)
    packed,_=lower(Graph.from_metadata(graph.metadata()),policy={'moe':dict(packed=True,packed_minimum_rows=1)})
    assert [c.kind for c in packed.calls]==['expert_pack','fp8_expert_swiglu','expert_partial']
    outputs=[]
    with tensor.Device() as device,ExitStack() as lifetime:
        buffers={name:device.from_numpy(array,dtype=graph.buffers[name].dtype) for name,array in host.items()}
        buffers['out']=device.zeros(out.shape,'bfloat16')
        for value in buffers.values():lifetime.callback(value.release)
        for name,candidate in [('original',graph),('packed',packed)]:
            destination=tmp_path/name;produce(candidate,destination,search=False,target='sm_89')
            with Forward(destination,device,buffers,graphs=True) as forward:
                forward.run();outputs.append(buffers['out'].to_numpy())
        np.testing.assert_array_equal(outputs[0],outputs[1])


@pytest.mark.skipif(os.getenv('AKAS_LFM_FP8_CUDA') != '1', reason='CUDA packed model qualification')
def test_packed_model_reorders_ragged_requests_and_reuses_state(tmp_path):
    from test_lfm_moe_native import test_native_model_reordered_ragged_requests_reuse_common_executor
    test_native_model_reordered_ragged_requests_reuse_common_executor(
        tmp_path,32,True,True,1,'bfloat16','bfloat16',packed_experts=True)


@pytest.mark.parametrize('guard',[None,'feature','alias','other_user'])
def test_packing_policy_preserves_public_or_shared_activations(tmp_path,guard):
    from test_lfm_moe_native import fixture
    from akas.models.lfm2 import LFM2MoE
    from akas.compilation.policy import lower
    from akas.compilation.compile import plan

    checkpoint=LFM2MoE.checkpoint_type(fixture(tmp_path/'weights'),tokenizer=False)
    graph=LFM2MoE.compose(checkpoint,rows=2,chunk=4,max_requests=4,max_context=64).graph
    lower(graph,policy={'moe':dict(fused_swiglu=True)})
    pair=next(c for c in graph.calls if c.kind=='fp8_expert_swiglu')
    name=pair.bindings['out'];oldshape=graph.buffers[name].shape
    if guard=='feature':graph.request['features']['encoded']=name
    elif guard=='alias':graph.buffers[name].view(oldshape)
    elif guard=='other_user':graph('unregistered_consumer',{},graph.buffers[name])
    lower(graph,policy={'moe':dict(packed=True,packed_minimum_rows=1)})
    assert sum(c.kind=='expert_pack' for c in graph.calls)==int(guard is None)
    if guard is None:plan(graph,target='sm_89')
    else:assert graph.buffers[name].shape==oldshape


@pytest.mark.parametrize('settings',[dict(packed=1),dict(packed=True,packed_minimum_rows=0),dict(packed=True,packed_minimum_rows=True),dict(packed=True,partitions=2)])
def test_packing_policy_rejects_invalid_contracts(settings):
    from akas.graph import Graph
    from akas.compilation.policy import lower
    with pytest.raises(ValueError,match='packing'):
        lower(Graph(contract='reference'),policy={'moe':settings})
