"""Host sampling is an explicit diagnostic over declared output features."""

from types import SimpleNamespace
import numpy as np
import pytest
from akas.generation.greedy import generate_scheduled


@pytest.mark.parametrize("sampling",["device","host"])
def test_sampling_policies_release_request_and_choose_same_lowest_tie(sampling):
    actions=[]
    feature=SimpleNamespace(to_numpy=lambda:np.array([0,2,2,0],dtype="float32"))
    frame=SimpleNamespace(features={"logits":"scores"},rows=1,
                          feature=lambda name:feature,
                          forward=SimpleNamespace(definition=SimpleNamespace(buffers={"scores":SimpleNamespace(role="output",shape=(4,))})))
    model=SimpleNamespace(context=32,config=SimpleNamespace(vocab=4),prefill_geometry=lambda:4,
                          batches=SimpleNamespace(select=lambda *args:frame),
                          admit=lambda:actions.append("admit") or 3,
                          release=lambda slot:actions.append(("release",slot)),
                          prefill=lambda *args,**kwargs:[1],decode=lambda *args:[1])
    tokenizer=SimpleNamespace(chat=lambda value:[2,3],encode=lambda value:[2,3],
                              decode=lambda ids:str(ids),eos=7)
    result=generate_scheduled(model,"text",tokenizer=tokenizer,max_tokens=3,sampling=sampling)
    assert result["generated_tokens"] == [1,1,1]
    assert actions == ["admit",("release",3)]


def test_host_sampling_rejects_undeclared_feature_before_admission():
    frame=SimpleNamespace(features={})
    model=SimpleNamespace(prefill_geometry=lambda:4,
                          batches=SimpleNamespace(select=lambda *args:frame),
                          admit=lambda:pytest.fail("invalid policy admitted request"))
    with pytest.raises(ValueError,match="declared logits"):
        generate_scheduled(model,"text",tokenizer=object(),sampling="host")
