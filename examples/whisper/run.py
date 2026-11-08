"""Reproduce an existing Whisper repair; not an independently discovered fix.

Repair algebra and neuron-selection recipe reimplement Kevin Bondzio's fit_gate:
https://github.com/kevinpl07/weight-patching/blob/d7df0d9bd15afb952b1e0cb64a067806563aa8e3/tasks/whisper_silence.py
Upstream MIT licence: UPSTREAM-LICENSE. Our data setup, development sweep and
control refinements do not establish a novel repair method. The measured run's
code revision is recorded separately in the case study and results.
"""
import json,time,hashlib,os,sys,random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import core as W

def acceptance(base,patch):
 return bool(patch['hallucination']<=base['hallucination']*.25 and base['hallucination']-patch['hallucination']>=.1 and patch['speech_kept']>=.99 and patch['speech_wer']['wer']<=base['speech_wer']['wer']+.5)

from safetensors.torch import save_file
ROOT=Path(os.environ.get('WHISPER_STUDY_DIR',str(Path(__file__).parent)));OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
torch.set_num_threads(4);torch.manual_seed(2026);np.random.seed(2026);random.seed(2026)
def save(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2,default=str)+'\n')
items=json.loads((ROOT/'manifest.json').read_text())
for item in items:assert hashlib.sha256((ROOT/'data'/item['audio']).read_bytes()).hexdigest()==item['sha256']
assert len({r['id'] for r in items})==len(items)
for split in ['diagnostic','development','test']:
 for other in ['diagnostic','development','test']:
  if split!=other:assert not ({r['group'] for r in items if r['split']==split}&{r['group'] for r in items if r['split']==other})
save('run.json',dict(start=time.time(),seed=2026,code_sha=os.environ.get('STUDY_CODE_SHA'),model_revision='973afd24965f72e36ca33b3055d56a652f456b4d',data_sha256=hashlib.sha256((ROOT/'manifest.json').read_bytes()).hexdigest(),torch=torch.__version__,gpu=torch.cuda.get_device_name(),backward_passes=0,supervision='Mean activation contrast from labelled diagnostic clips; no model optimizer'))
proc,model=W.load('openai/whisper-small');W.gpu_setup();model.to('cuda');tok=W.Tok(model,proc);P=f'model.decoder.layers.{model.config.decoder_layers-1}.';params=dict(model.named_parameters());layer=model.model.decoder.layers[-1]
def run(split,tag,capture=False,hook=None):
 selected=[r for r in items if r['split']==split];rows=[];features=[]
 handle=layer.final_layer_norm.register_forward_hook(hook) if hook else None
 try:
  for offset in range(0,len(selected),8):
   part=selected[offset:offset+8];aud=[np.load(ROOT/'data'/r['audio']) for r in part]
   result=W.transcribe(model,proc,tok,aud,batch=8,capture=capture)
   for meta,r in zip(part,result):
    r=dict(r)
    if capture:features.append(r.pop('x'))
    rows.append({**meta,**r})
   print(tag,offset+len(part),'/',len(selected),flush=True)
 finally:
  if handle:handle.remove()
 save(tag+'.json',rows)
 return rows,features

def score(rows,base=None):
 ns=[r for r in rows if r['kind']=='nonspeech'];sp=[r for r in rows if r['kind']=='speech']
 out={'hallucination':np.mean([bool(W.norm(r['text'])) for r in ns]),'speech_wer':W.word_stats([r['ref'] for r in sp],[r['text'] for r in sp]),'n_nonspeech':len(ns),'n_speech':len(sp)}
 if base:
  lookup={r['id']:r for r in base};out['speech_kept']=np.mean([W.norm(r['text'])==W.norm(lookup[r['id']]['text']) for r in sp])
 return out

start=time.time()
# Check our inspected greedy implementation against the library's generation path.
pilot=next(r for r in items if r['split']=='diagnostic' and r['kind']=='speech')
audio=np.load(ROOT/'data'/pilot['audio']);feats=W.features(proc,[audio],'cuda')
with torch.inference_mode():
 reference=model.generate(input_features=feats,language='en',task='transcribe',return_timestamps=False,do_sample=False,num_beams=1,max_new_tokens=tok.max_new)
expected=proc.tokenizer.batch_decode(reference,skip_special_tokens=True)[0]
actual=W.transcribe(model,proc,tok,[audio],batch=1)[0]['text']
save('parity.json',{'reference':expected,'custom':actual,'normalized_equal':W.norm(expected)==W.norm(actual)})
assert W.norm(expected)==W.norm(actual),'Greedy decoder parity failed'
diag,X=run('diagnostic','diagnostic_base',True);save('diagnostic_metrics.json',score(diag))
if score(diag)['hallucination']<.25:save('verdict.json',{'status':'failure_gate_not_met'});sys.exit(0)
ns=[x[3] for r,x in zip(diag,X) if r['kind']=='nonspeech'];sp=[x[3] for r,x in zip(diag,X) if r['kind']=='speech'];d=torch.stack(ns).mean(0)-torch.stack(sp).mean(0);d=d/d.norm()
nsproj=torch.stack(ns)@d;allsp=torch.cat([x[3:] for r,x in zip(diag,X) if r['kind']=='speech'])@d
threshold=(float(allsp.max())+float(nsproj.min()))/2
# Dose/causal controls change only the first content-position MLP input.
for tag,sign in [('remove_signal',-1),('inject_signal',1),('sham',0),('random_direction',1)]:
 random_d=torch.randn(d.shape,generator=torch.Generator().manual_seed(2026));random_d/=random_d.norm()
 direction=random_d if tag=='random_direction' else d
 def hook(mod,args,out,sign=sign,direction=direction):
  if out.shape[1]!=4:return
  out=out.clone();dd=direction.to(out.device);out[:,3]+=sign*4*dd;return out
 rows,_=run('diagnostic',tag,hook=hook)
 save(tag+'-metrics.json',score(rows,diag))
w1=params[P+'fc1.weight'].detach().cpu().clone();b1=params[P+'fc1.bias'].detach().cpu().clone();w2=params[P+'fc2.weight'].detach().cpu().clone()
use=torch.zeros(len(b1))
for xx in X:use+=F.gelu(xx@w1.T+b1).abs().sum(0)
j=int((use*w2.norm(dim=0)).argmin());dot=tok.encode(' .')[0]
competitors=sorted({t for r in diag if r['kind']=='nonspeech' for t in r['top_first'][:5] if t<tok.eot and t!=dot})
E=params['model.decoder.embed_tokens.weight'].detach().cpu();v=(E[dot]-E[competitors].mean(0))/params['model.decoder.layer_norm.weight'].detach().cpu();v=v/v.norm()
original={P+'fc1.weight':w1,P+'fc1.bias':b1,P+'fc2.weight':w2}
@torch.no_grad()
def apply(gain,direction=d,write=v,cutoff=None):
 for name,value in original.items():params[name].copy_(value.to('cuda'))
 params[P+'fc1.weight'][j].copy_(direction.to('cuda'));params[P+'fc1.bias'][j]=-(threshold if cutoff is None else cutoff);params[P+'fc2.weight'][:,j].copy_((gain*write).to('cuda'))
@torch.no_grad()
def restore():
 for name,value in original.items():params[name].copy_(value.to('cuda'))
dev,_=run('development','development_base');baseline=score(dev);candidates=[]
for cutoff in [threshold,float(allsp.max())+1,float(allsp.max())+4]:
 for gain in [2.,4.,8.]:
  apply(gain,cutoff=cutoff);rows,_=run('development','development_gain_'+str(gain)+'_cut_'+str(round(cutoff,3)));s=score(rows,dev);candidates.append({'gain':gain,'threshold':cutoff,**s})
restore();valid=[c for c in candidates if c['speech_kept']>=.99 and c['speech_wer']['wer']<=baseline['speech_wer']['wer']+.5];best=min(valid,key=lambda c:c['hallucination']) if valid else None
save('selection.json',{'layer':model.config.decoder_layers-1,'neuron':j,'threshold':threshold,'ns_projection':nsproj.tolist(),'speech_projection_max':float(allsp.max()),'candidates':candidates,'selected':best})
base,_=run('test','test_base');metrics={'base':score(base)}
for label,cfg in [('default',W.OPENAI_DEFAULT)]+[(f'threshold_{t}',{'no_speech':t,'logprob':None}) for t in [.1,.2,.4,.6,.8]]:
 # Threshold selection uses development only.
 tuned=[{**r,'text':r['text'] if W.threshold_keep(r,cfg) else ''} for r in dev];s=score(tuned,dev)
 if label=='default' or (s['speech_kept']>=.99 and s['speech_wer']['wer']<=baseline['speech_wer']['wer']+.5):
  rows=[{**r,'text':r['text'] if W.threshold_keep(r,cfg) else ''} for r in base];metrics[label]=score(rows,base);save('test_'+label+'.json',rows)
if best:
 threshold=best['threshold'];apply(best['gain']);patched,_=run('test','test_patch');metrics['patch']=score(patched,base)
 patch={k:params[k].detach().cpu().contiguous() for k in original};save_file(patch,str(OUT/'patch.safetensors'))
 save('patch_manifest.json',{'changed_entries':sum(int((patch[k]!=original[k]).sum()) for k in original),'total_parameters':sum(p.numel() for p in model.parameters()),'layer':P,'neuron':j,'gain':best['gain'],'threshold':threshold})
 restore()
 # Re-load the checkpoint and reapply the frozen patch, then reproduce sampled outputs.
 fresh_proc,fresh=W.load('openai/whisper-small');fresh.to('cuda')
 with torch.no_grad():
  fp=dict(fresh.named_parameters())
  for name,value in patch.items():fp[name].copy_(value.to('cuda'))
 sample=[r for r in items if r['split']=='test'][:8]
 check=W.transcribe(fresh,fresh_proc,W.Tok(fresh,fresh_proc),[np.load(ROOT/'data'/r['audio']) for r in sample],batch=8)
 lookup={r['id']:r for r in patched}
 save('fresh_reload.json',{'examples':len(sample),'tokens_equal':all(a['tokens']==lookup[b['id']]['tokens'] for a,b in zip(check,sample))})
 del fresh;torch.cuda.empty_cache()
 for seed in range(3):
  g=torch.Generator().manual_seed(seed)
  with torch.no_grad():
   for name,index,desired in [(P+'fc1.weight',(j,slice(None)),d),(P+'fc2.weight',(slice(None),j),best['gain']*v)]:
    old=original[name][index];noise=torch.randn(old.shape,generator=g);noise=noise/noise.norm()*(desired-old).norm();params[name][index].copy_((old+noise).to('cuda'))
   params[P+'fc1.bias'][j]=-threshold
  rows,_=run('test','test_random_'+str(seed));metrics['random_'+str(seed)]=score(rows,base);restore()
 save('verdict.json',{'status':'evaluated','patch_passes_point_estimates':acceptance(metrics['base'],metrics['patch']),'restored':all(torch.equal(params[k].detach().cpu(),v) for k,v in original.items())})
else:save('verdict.json',{'status':'no_development_candidate_passed_preservation'})
save('metrics.json',metrics);save('runtime.json',{'seconds':time.time()-start});print('STUDY_COMPLETE',flush=True)
