import importlib.util
from pathlib import Path
import torch
spec=importlib.util.spec_from_file_location('case',Path(__file__).resolve().parents[1]/'examples/robustmnist_case.py')
case=importlib.util.module_from_spec(spec);spec.loader.exec_module(case)

def test_split_variants_cannot_cross_source_boundaries():
    splits=case.split_ids()
    sets=[set(v.tolist()) for v in splits.values()]
    assert [len(s) for s in sets]==[2000,2000,6000]
    assert len(set.union(*sets))==10000
    assert not any(a&b for i,a in enumerate(sets) for b in sets[i+1:])

def test_translation_has_no_wraparound_and_identity_is_exact():
    x=torch.zeros(1,1,28,28);x[0,0,14,14]=1
    assert torch.equal(case.transform(x,'clean'),x)
    assert case.transform(x,'x:2')[0,0,14,16]==1
    assert case.transform(x,'y:-2')[0,0,12,14]==1
    assert torch.equal(case.transform(case.transform(x,'x:2'),'x:-2'),x)
    x.zero_();x[0,0,14,27]=1
    assert case.transform(x,'x:2').sum()==0

def test_unknown_is_not_counted_as_confident_digit_error():
    scores=torch.zeros(2,11);scores[0,10]=20;scores[1,2]=20
    result=case.metrics(scores,torch.tensor([1,1]))
    assert result['accuracy']==0 and result['unknown']==.5 and result['confident_wrong']==.5

def test_batched_predictions_keep_every_sample():
    model=torch.nn.Flatten()
    x=torch.arange(131.).reshape(131,1)
    assert torch.equal(case.predict(model,x),x)


def test_repair_requires_gain_uncertainty_and_both_preservation_gates():
    assert case.accept_repair(.06,.01,.001,.01)
    for args in [(.04,.01,0,0),(.06,0,0,0),(.06,.01,.006,0),(.06,.01,0,.03)]:
        assert not case.accept_repair(*args)
