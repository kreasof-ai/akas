"""Attention partition scratch follows a workload-derived split budget."""
import pytest
from akas.graph import Graph
from akas.compilation.policy import lower


@pytest.mark.parametrize('rows,expected',[(1,16),(4,8),(16,2),(64,1),(256,1)])
def test_grouped_decode_split_budget_updates_producer_consumer_and_scratch(rows,expected):
    graph=Graph(contract='reference')
    partial=graph.buffer('partial',(rows*32*16*66,),'float32')
    p=dict(r=rows,h=32,kh=8,d=64,capacity=2176,splits=16,optimized=True,
           kv_dtype='float16',unit_scales=True,provider='cuda')
    graph('attention_partial',p,out=partial)
    graph('attention_merge',p,partial=partial)
    lower(graph,policy={'attention':dict(decode_precision='float16',splits=16,minimum_split_groups=256)})
    assert [call.parameters['splits'] for call in graph.calls]==[expected,expected]
    assert graph.calls[0].parameters['decode_precision']=='float16'
    assert graph.buffers['partial'].shape==(rows*32*expected*66,)


@pytest.mark.parametrize('value',[True,0,-1,1.5])
def test_attention_split_budget_rejects_invalid_counts(value):
    with pytest.raises(ValueError,match='positive CUDA workgroup count'):
        lower(Graph(contract='reference'),policy={'attention':dict(minimum_split_groups=value)})


def test_split_budget_keeps_an_explicit_partition_cap():
    graph=Graph(contract='reference');partial=graph.buffer('partial',(32*2*66,),'float32')
    p=dict(r=1,h=32,kh=8,d=64,splits=2,optimized=True)
    graph('attention_partial',p,out=partial)
    lower(graph,policy={'attention':dict(decode_precision='float16',minimum_split_groups=256)})
    assert graph.calls[0].parameters['splits']==2
    assert graph.buffers['partial'].shape==(32*2*66,)


@pytest.mark.parametrize('precision',['float32','float16'])
def test_split_budget_counts_kv_groups_independently_of_operand_precision(precision):
    graph=Graph(contract='reference');partial=graph.buffer('partial',(16*32*16*66,),'float32')
    p=dict(r=16,h=32,kh=8,d=64,splits=16,optimized=True)
    graph('attention_partial',p,out=partial)
    lower(graph,policy={'attention':dict(decode_precision=precision,minimum_split_groups=512)})
    assert graph.calls[0].parameters['splits']==4
    assert graph.buffers['partial'].shape==(16*32*4*66,)
