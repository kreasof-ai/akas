"""Phase policy selection reuses Tensor's existing parameter-profile rules."""
import pytest
from akas.compilation.policy import lower,resolve
from akas.graph import Graph


def profile(entries):
    from tensor.compiler.search import ScheduleProfile
    return ScheduleProfile(dict(schema='tensor.schedule-profile.v1',provider='cuda',target='sm_89',entries=entries))


def entry(parameters,settings,operation='policy.attention'):
    return dict(operation=operation,parameters=parameters,schedule=settings)


@pytest.mark.parametrize('phase,rows,expected',[('decode',64,1),('decode',4,16),('prefill',64,16)])
def test_phase_override_preserves_defaults_and_resizes_the_existing_graph(phase,rows,expected):
    defaults={'attention':dict(splits=16,decode_precision='float32')}
    selected=resolve(defaults,profile([entry(dict(phase='decode',rows=64),dict(splits=1))]),
                     dict(phase=phase,rows=rows,chunk=1),provider='cuda',target='sm_89')
    assert defaults['attention']['splits']==16
    assert selected['attention']==dict(splits=expected,decode_precision='float32')
    graph=Graph(contract='reference');partial=graph.buffer('partial',(rows*32*16*66,),'float32')
    p=dict(r=rows,h=32,kh=8,d=64,splits=16,optimized=True)
    graph('attention_partial',p,out=partial);graph('attention_merge',p,partial=partial)
    lower(graph,policy=selected)
    assert [c.parameters['splits'] for c in graph.calls]==[expected,expected]
    assert graph.buffers['partial'].shape==(rows*32*expected*66,)


def test_kernel_entries_do_not_become_operation_policies():
    defaults={'moe':dict(fused_swiglu=True)}
    assert resolve(defaults,profile([entry({},dict(implementation='attention.cuda.floating'),'attention_partial')]),
                   {},provider='cuda',target='sm_89')==defaults


def test_profile_rejects_unknown_policy_families():
    with pytest.raises(ValueError,match='unknown operation policies'):
        resolve({},profile([entry({},dict(enabled=True),'policy.unregistered')]),{},provider='cuda',target='sm_89')


def test_profile_keeps_existing_ambiguity_checks():
    choices=profile([entry(dict(phase='decode'),dict(splits=1)),entry(dict(rows=64),dict(splits=8))])
    with pytest.raises(ValueError,match='ambiguous'):
        resolve({},choices,dict(phase='decode',rows=64),provider='cuda',target='sm_89')


def test_profile_preserves_policy_order_and_explicit_false_values():
    defaults={'moe':dict(fused_swiglu=True),'linear':dict(stage_inputs=True)}
    choices=profile([entry({},dict(fused_swiglu=False),'policy.moe'),entry({},dict(splits=4))])
    selected=resolve(defaults,choices,{},provider='cuda',target='sm_89')
    assert list(selected)==['moe','linear','attention']
    assert selected['moe']['fused_swiglu'] is False


def test_profile_requires_matching_target():
    with pytest.raises(ValueError,match='provider/target mismatch'):
        resolve({},profile([entry({},dict(splits=4))]),{},provider='cuda',target='sm_90')
