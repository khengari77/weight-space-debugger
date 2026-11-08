"""Pinned real-model investigation. Upstream architecture: Apache-2.0 (MultivexAI).

Downloads are cached outside the package. No training or backward passes.
"""
from __future__ import annotations
import argparse, gzip, hashlib, importlib.util, json, math, sys, time, urllib.request
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from causal import capture_activation, inject_activation
from patching import TensorEdit, patch_state, forward_with_patch
from probe import ModuleProbe

REV = '6742662f0e48a88b7d069d2094d211f8fd106d5a'
MODEL = 'https://huggingface.co/MultivexAI/RobustMNIST-v1.0'
HASH = '9fb6645f700d0bcc50cd76f6ec8b31e031d777fe8634b85d2a162406eee482d7'
CODE_HASH = '4ca557a212f5d7653ddfe80b753e2f5f5419b2b0a688579bdbce3cc5ce2ac043'

def fetch(cache, name, url):
    path = cache / name
    if not path.exists():
        urllib.request.urlretrieve(url, path)
    return path

def dataset(cache, fashion=False):
    base = ('https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/'
            if fashion else 'https://ossci-datasets.s3.amazonaws.com/mnist/')
    prefix = 'fashion-' if fashion else ''
    images = gzip.decompress(fetch(cache, prefix+'images.gz', base+'t10k-images-idx3-ubyte.gz').read_bytes())
    labels = gzip.decompress(fetch(cache, prefix+'labels.gz', base+'t10k-labels-idx1-ubyte.gz').read_bytes())
    return torch.from_numpy(np.frombuffer(images, np.uint8, offset=16).copy()).reshape(-1,1,28,28).float()/255, torch.from_numpy(np.frombuffer(labels,np.uint8,offset=8).copy()).long()

def transform(x, kind):
    if kind == 'clean': return x.clone()
    direction, value = kind.split(':'); value = int(value)
    if direction in ('x','y'):
        out = torch.zeros_like(x)
        axis = 3 if direction == 'x' else 2
        source = [slice(None)]*4; target = source.copy()
        source[axis] = slice(None,-value) if value > 0 else slice(-value,None)
        target[axis] = slice(value,None) if value > 0 else slice(None,value)
        out[tuple(target)] = x[tuple(source)]
        return out
    a = math.radians(value)
    theta = x.new_tensor([[math.cos(a),-math.sin(a),0],[math.sin(a),math.cos(a),0]])
    return F.grid_sample(x,F.affine_grid(theta.expand(len(x),-1,-1),x.shape,align_corners=False),align_corners=False)

def center(x):
    out = torch.zeros_like(x)
    coords = torch.arange(28).float()
    for i, image in enumerate(x):
        mass = image.sum()
        if mass == 0: continue
        dx = round(13.5-float((image.sum(1).squeeze()*coords).sum()/mass))
        dy = round(13.5-float((image.sum(2).squeeze()*coords).sum()/mass))
        moved = image.unsqueeze(0)
        for axis, value in [('x',dx),('y',dy)]:
            if value: moved = transform(moved, f'{axis}:{value}')
        out[i] = moved[0]
    return out

def split_ids():
    ids = torch.randperm(10000,generator=torch.Generator().manual_seed(2026))
    return {'diagnostic':ids[:2000], 'development':ids[2000:4000], 'sealed':ids[4000:]}

@torch.inference_mode()
def predict(model, x, state=None):
    return torch.cat([(model(batch) if state is None else forward_with_patch(model,state,batch)).cpu() for batch in x.split(64)])

def metrics(logits,y):
    p = logits.softmax(-1); pred = p.argmax(-1)
    return {'accuracy':float((pred==y).float().mean()),'unknown':float((pred==10).float().mean()),
            'confident_wrong':float(((pred!=y)&(pred!=10)&(p.max(-1).values>=.9)).float().mean()),'count':len(y), 'per_class_accuracy':{str(int(c)):float((pred[y==c]==y[y==c]).float().mean()) for c in y.unique()}}

def accept_repair(gain, ci_low, clean_drop, rejection_drop):
    return gain >= .05 and ci_low > 0 and clean_drop <= .005 and rejection_drop <= .02

def save(out,name,data):
    (out/name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--phase',choices=['reproduce','investigate','evaluate'],required=True)
    ap.add_argument('--cache-dir',type=Path,default=Path('/tmp/robustmnist-case'))
    ap.add_argument('--output',type=Path,default=Path('reports/robustmnist'))
    args=ap.parse_args(); cache=args.cache_dir; out=args.output
    cache.mkdir(parents=True,exist_ok=True);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.manual_seed(2026); start=time.monotonic()
    weights=fetch(cache,'model.safetensors',MODEL+'/resolve/'+REV+'/model.safetensors')
    code=fetch(cache,'model.py',MODEL+'/raw/'+REV+'/model.py')
    assert hashlib.sha256(weights.read_bytes()).hexdigest()==HASH
    assert hashlib.sha256(code.read_bytes()).hexdigest()==CODE_HASH
    # This exact source was manually inspected before execution; reject revisions.
    spec=importlib.util.spec_from_file_location('reviewed_robustmnist',code)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    model=module.HierarchicalNetwork().eval(); model.load_state_dict(load_file(weights))
    base={k:v.clone() for k,v in model.state_dict().items()}
    passes={'forward_batches':0,'input_examples':0}
    def count_passes(_model, inputs):
        passes['forward_batches']+=1; passes['input_examples']+=len(inputs[0])
    model.register_forward_pre_hook(count_passes)
    x,y=dataset(cache); ids=split_ids()
    if args.phase=='reproduce':
        ix=ids['diagnostic']; clean=predict(model,x[ix]); result={'clean':metrics(clean,y[ix]),'transforms':{}}
        kinds=[f'{axis}:{v}' for v in (2,-2,4,-4) for axis in ('x','y')]+['rotation:15','rotation:-15']
        for kind in kinds:
            logits=predict(model,transform(x[ix],kind)); flips=int(((clean.argmax(-1)==y[ix])&(logits.argmax(-1)!=y[ix])).sum())
            result['transforms'][kind]={**metrics(logits,y[ix]),'clean_correct_to_wrong':flips}
            print(kind,result['transforms'][kind],flush=True)
        candidates=[k for k in kinds if result['transforms'][k]['clean_correct_to_wrong']>=50]
        if not candidates:
            for kind in ['rotation:30','rotation:-30']:
                logits=predict(model,transform(x[ix],kind))
                result['transforms'][kind]={**metrics(logits,y[ix]),'clean_correct_to_wrong':int(((clean.argmax(-1)==y[ix])&(logits.argmax(-1)!=y[ix])).sum())}
                if result['transforms'][kind]['clean_correct_to_wrong']>=50:candidates.append(kind)
        result['selected']=candidates[0] if candidates else None
        result['revision']=REV;result['checkpoint_sha256']=HASH;result['parameters']=sum(p.numel() for p in model.parameters())
        save(out,'reproduction.json',result)
    else:
        reproduction=json.loads((out/'reproduction.json').read_text()); kind=reproduction['selected']
        if kind is None:
            if args.phase == 'investigate':
                save(out,'investigation.json',{'verdict':'negative_replication','candidate':None,'reason':'No diagnostic condition reached 50 clean-correct to transformed-wrong cases; causal localization and surgery not justified.'})
            else:
                ix=ids['sealed']; sx=x[ix]; sy=y[ix]
                result={'verdict':'negative_replication','candidate':None,'accepted':False,'clean':metrics(predict(model,sx),sy),'transforms':{}}
                for condition in ['rotation:30','rotation:-30']:
                    result['transforms'][condition]=metrics(predict(model,transform(sx,condition)),sy)
                fx,_=dataset(cache,True)
                result['ood_rejection']=float((predict(model,fx[:2000]).argmax(-1)==10).float().mean())
                result['original_model_unchanged']=all(torch.equal(v,base[k]) for k,v in model.state_dict().items())
                save(out,'evaluation.json',result)
            save(out,args.phase+'-runtime.json',{'seconds':time.monotonic()-start,'torch':torch.__version__,'seed':2026,'backward_passes':0,**passes})
            return
        if args.phase=='investigate':
            ix=ids['diagnostic']; clean_x=x[ix]; corrupt=transform(clean_x,kind)
            clean=predict(model,clean_x); before=predict(model,corrupt)
            failing=((clean.argmax(-1)==y[ix])&(before.argmax(-1)!=y[ix])).nonzero().flatten()[:32]
            sites=['pre_processor','stage_a.refinement.activation','stage_b.refinement.activation','stage_c.refinement.activation','global_reducer_a','global_reducer_b']
            ledger={'hypotheses':[
                {'id':'H1','prediction':'Input centering restores shifted inputs without changing weights'},
                {'id':'H2','prediction':'Clean spatial gates rescue errors; inverse transfer harms clean predictions'},
                {'id':'H3','prediction':'Late pooled features rescue errors even if spatial gates do not'},
                {'id':'null','prediction':'Random or sham interventions rescue equally well'}], 'selected':kind,'sites':{}}
            a=clean_x[failing];b=corrupt[failing];target=y[ix][failing]
            with torch.inference_mode(),ModuleProbe(model,['pre_processor','stage_a','stage_b','stage_c'],max_events=1) as probe:
                model(b)
            ledger['probe']=probe.summary()
            ledger['failure_source_ids']=ix[failing].tolist()
            for site in sites:
                _, ca=capture_activation(model,site,a);_, cb=capture_activation(model,site,b)
                results={}
                for label,inputs,source in [('rescue',b,ca),('sham',b,cb),('reverse',a,cb),('random_pair',b,ca.roll(1,0))]:
                    with torch.inference_mode(),inject_activation(model,site,source): logits=model(inputs)
                    results[label]=metrics(logits,target)
                ledger['sites'][site]=results
            dev=ids['development'];dx=x[dev];dy=y[dev];cx=transform(dx,kind)
            ledger['development_base']=metrics(predict(model,cx),dy)
            ledger['development_centering']=metrics(predict(model,center(cx)),dy)
            # Gate edits are tried only where matched interventions outperform sham/random.
            eligible=[s for s in sites if 'refinement.activation' in s and ledger['sites'][s]['rescue']['accuracy']>ledger['sites'][s]['random_pair']['accuracy']+.1 and ledger['sites'][s]['rescue']['accuracy']>ledger['sites'][s]['sham']['accuracy']+.1]
            candidates=[]
            for site in eligible:
                name=site.replace('.activation','.spatial_weighting.weight')
                for scale in [0.,.5,.75,1.25]:
                    state,manifest=patch_state(model,[TensorEdit(name,Ellipsis,'scale',scale)])
                    candidates.append({'name':name,'scale':scale,'manifest':manifest,'corrupt':metrics(predict(model,cx,state),dy),'clean':metrics(predict(model,dx,state),dy)})
            clean_dev=metrics(predict(model,dx),dy)
            valid=[c for c in candidates if c['corrupt']['accuracy']>=ledger['development_base']['accuracy']+.05 and c['clean']['accuracy']>=clean_dev['accuracy']-.005]
            best=max(valid,key=lambda c:c['corrupt']['accuracy']) if valid else None
            ledger['candidates']=candidates;ledger['candidate']=best;ledger['baseline_clean']=clean_dev
            ledger['baseline_failed_accuracy']=metrics(before[failing],target)
            save(out,'investigation.json',ledger)
        else:
            ledger=json.loads((out/'investigation.json').read_text());ix=ids['sealed'];sx=x[ix];sy=y[ix];cx=transform(sx,kind)
            result={'selected':kind,'clean':metrics(predict(model,sx),sy),'corrupt':metrics(predict(model,cx),sy),'centering':metrics(predict(model,center(cx)),sy),'candidate':ledger['candidate'],'accepted':False}
            fx,_=dataset(cache,True);fx=fx[:2000]
            result['ood_base']=float((predict(model,fx).argmax(-1)==10).float().mean())
            if ledger['candidate']:
                from metrics import paired_bootstrap_difference
                c=ledger['candidate']; state,manifest=patch_state(model,[TensorEdit(c['name'],Ellipsis,'scale',c['scale'])])
                before=predict(model,cx);after=predict(model,cx,state)
                result['patched']=metrics(after,sy);result['patched_clean']=metrics(predict(model,sx,state),sy)
                result['interval']=paired_bootstrap_difference((before.argmax(-1)==sy).float().tolist(),(after.argmax(-1)==sy).float().tolist())
                fx,_=dataset(cache,True);fx=fx[:2000]
                result['ood_base']=float((predict(model,fx).argmax(-1)==10).float().mean());result['ood_patch']=float((predict(model,fx,state).argmax(-1)==10).float().mean())
                # Null edits have the candidate's exact support and norm.
                original=dict(model.named_parameters())[c['name']].detach();delta=state[c['name']]-original;controls=[]
                for seed in range(10):
                    g=torch.Generator().manual_seed(seed);noise=torch.randn(original.shape,generator=g);noise=noise/noise.norm()*delta.norm()
                    null,_=patch_state(model,[TensorEdit(c['name'],Ellipsis,'add',noise)])
                    controls.append(metrics(predict(model,cx,null),sy))
                result['random_controls']=controls
                result['accepted']=accept_repair(result['patched']['accuracy']-result['corrupt']['accuracy'], result['interval']['percentile_bootstrap_ci_low'], result['clean']['accuracy']-result['patched_clean']['accuracy'], result['ood_base']-result['ood_patch'])
            result['original_model_unchanged']=all(torch.equal(v,base[k]) for k,v in model.state_dict().items())
            save(out,'evaluation.json',result)
    save(out,args.phase+'-runtime.json',{'seconds':time.monotonic()-start,'torch':torch.__version__,'seed':2026,'backward_passes':0,**passes})
if __name__=='__main__':main()
