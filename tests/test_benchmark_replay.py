"""Reference evidence rejects shape, provenance and nonfinite ambiguity."""

from types import SimpleNamespace
import numpy as np
import pytest

from akas.benchmarking.replay import evaluate, metrics


def test_zero_reference_and_nonfinite_output_are_unambiguous():
    result=metrics(np.zeros(3),np.zeros(3))
    assert result["relative_rms"] == 0 and result["cosine"] == 1
    assert not metrics(np.array([np.nan]),np.array([1.]))["finite"]
    with pytest.raises(ValueError,match="shape"):
        metrics(np.zeros(2),np.zeros(3))
    with pytest.raises(ValueError,match="nonfinite"):
        metrics(np.ones(1),np.array([np.inf]))


def test_forced_feature_replay_selects_decode_and_validates_before_mutation():
    calls=[]
    session=SimpleNamespace(config=SimpleNamespace(vocab=8),batch=1,slots=[3],
                            executor=SimpleNamespace(context=8,positions=[0,0,0,4]),
                            step=lambda v:calls.append(("decode",v.tolist())),
                            prefill=lambda v:calls.append(("prefill",v[0].tolist())),
                            read_feature=lambda name:np.array([[1,2,3]]))
    np.testing.assert_array_equal(evaluate(session,[4]),[1,2,3])
    evaluate(session,[1,2])
    assert calls == [("decode",[4]),("prefill",[1,2])]
    for values, reason in (([],"nonempty"),([8],"vocabulary"),([1]*5,"capacity")):
        with pytest.raises(ValueError,match=reason):
            evaluate(session,values)
    assert len(calls) == 2
