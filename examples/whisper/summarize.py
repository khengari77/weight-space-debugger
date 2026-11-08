"""Summarize fixed Whisper study outputs; raw runs remain outside Git."""
import argparse,json,random
from pathlib import Path
import numpy as np
import core as W
ap=argparse.ArgumentParser();ap.add_argument('--results',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();root=args.results;out=args.output;out.mkdir(parents=True,exist_ok=True)
def read(name):return json.loads((root/name).read_text())
base=read('test_base.json');patch=read('test_patch.json');lookup={r['id']:r for r in patch};ns=[r for r in base if r['kind']=='nonspeech'];sp=[r for r in base if r['kind']=='speech']
groups=sorted({r['group'] for r in ns});flags={r['id']:int(bool(W.norm(r['text'])))-int(bool(W.norm(lookup[r['id']]['text']))) for r in ns};rng=random.Random(2026);diffs=[]
for _ in range(2000):
 sample=[r for g in rng.choices(groups,k=len(groups)) for r in ns if r['group']==g];diffs.append(np.mean([flags[r['id']] for r in sample]))
summary={'metrics':read('metrics.json'),'patch':read('patch_manifest.json'),'tensor_details':read('patch_detail.json'),'verdict':read('verdict.json'),'fresh_reload':read('fresh_reload.json'),'paired_improvement_ci95':np.quantile(diffs,[.025,.975]).tolist(),'bootstrap':{'unit':'original environmental recording','groups':len(groups),'replicates':2000,'seed':2026},'speech_unchanged_count':sum(W.norm(r['text'])==W.norm(lookup[r['id']]['text']) for r in sp),'speech_count':len(sp),'failure_count_before':sum(bool(W.norm(r['text'])) for r in ns),'failure_count_after':sum(bool(W.norm(lookup[r['id']]['text'])) for r in ns),'non_speech_count':len(ns),'run':read('run.json'),'runtime':read('runtime.json'),'selection':read('selection.json')}
summary['raw_speech_unchanged_count']=sum(r['text']==lookup[r['id']]['text'] for r in sp)
summary['provenance']={'classification': 'reproduction_with_implementation_and_evaluation_refinements', 'repair_method_source': 'https://github.com/kevinpl07/weight-patching/tree/d7df0d9bd15afb952b1e0cb64a067806563aa8e3/tasks/whisper_silence.py', 'decoder_source': 'https://github.com/kevinpl07/weight-patching/tree/d7df0d9bd15afb952b1e0cb64a067806563aa8e3/tasks/_whisper_core.py', 'independently_discovered_fix': False, 'novelty_assessment': 'No significant new repair method or methodological advance established.', 'inherited_methods': ['activation contrast and neuron-gate repair', 'decoder helpers', 'held-out evaluation', 'speech preservation checks', 'decoding baselines', 'random controls'], 'local_changes': ['smaller evaluation sample and data preparation', 'development strength/threshold sweep', 'actual-delta-norm control matching', 'additional verification and uncertainty estimates', 'Agent Skill packaging'], 'comparison_limit': 'No same-data upstream comparison or blind investigator benchmark; no superiority claim.'}
summary['accepted']=bool(summary['verdict']['patch_passes_point_estimates'] and summary['paired_improvement_ci95'][0]>0 and summary['fresh_reload']['tokens_equal'])
examples=[]
for r in ns:
 after=lookup[r['id']]['text']
 if W.norm(r['text']) and not W.norm(after) and len(r['text'])<160:
  examples.append({'id':r['id'],'category':r['category'],'before':r['text'],'after':after,'audio_url':'https://github.com/karolpiczak/ESC-50/blob/master/audio/'+r['id']})
 if len(examples)==4:break
summary['examples']=examples
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
metrics=summary['metrics'];labels=list(metrics)
# Re-sample original recordings, preserving all clips from each sampled source.
ng=len(groups);draws=np.random.default_rng(2026).integers(0,ng,size=(2000,ng));counts=np.array([sum(r['group']==g for r in ns) for g in groups]);intervals={}
for label in labels:
 arm=read('test_'+label+'.json');arm_lookup={r['id']:r for r in arm}
 totals=np.array([sum(bool(W.norm(arm_lookup[r['id']]['text'])) for r in ns if r['group']==g) for g in groups])
 samples=totals[draws].sum(1)/counts[draws].sum(1);intervals[label]=np.quantile(samples,[.025,.975]).tolist()
summary['hallucination_ci95']=intervals
baseline_default={r['id']:r for r in read('test_default.json')}
delta_totals=np.array([sum(int(bool(W.norm(baseline_default[r['id']]['text'])))-int(bool(W.norm(lookup[r['id']]['text']))) for r in ns if r['group']==g) for g in groups])
summary['default_vs_patch_improvement_ci95']=np.quantile(delta_totals[draws].sum(1)/counts[draws].sum(1),[.025,.975]).tolist()
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
rates=[100*metrics[k]['hallucination'] for k in labels];kept=[100*metrics[k].get('speech_kept',1) for k in labels]
fig,axes=plt.subplots(1,2,figsize=(12,4));axes[0].bar(labels,rates,yerr=np.array([[max(0,rates[i]-100*intervals[k][0]) for i,k in enumerate(labels)],[max(0,100*intervals[k][1]-rates[i]) for i,k in enumerate(labels)]]),capsize=3);axes[0].set(ylabel='Clips with invented words (%)',ylim=(0,105),title='200 sealed environmental clips');axes[1].bar(labels,kept);axes[1].set(ylabel='Speech transcripts unchanged (%)',ylim=(0,105),title='200 sealed speech clips')
for ax in axes:ax.tick_params(axis='x',rotation=55)
fig.tight_layout();fig.savefig(out/'comparison.png',dpi=150);plt.close(fig)
selection=summary['selection'];fig,ax=plt.subplots(figsize=(7,3));ax.hist(selection['ns_projection'],bins=12,label='Diagnostic non-speech');ax.axvline(selection['speech_projection_max'],color='green',label='Maximum diagnostic speech projection');ax.axvline(selection['selected']['threshold'],color='red',linestyle='--',label='Frozen gate threshold');ax.set(xlabel='Projection onto diagnostic contrast direction',ylabel='Clips');ax.legend(fontsize=7);fig.tight_layout();fig.savefig(out/'gate.png',dpi=150)
print(json.dumps({k:summary[k] for k in ['accepted','failure_count_before','failure_count_after','speech_unchanged_count','paired_improvement_ci95']},indent=2))
