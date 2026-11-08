"""Export the real-model diagnostic figure; requires matplotlib."""
import argparse, importlib.util, json, time
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from safetensors.torch import load_file
import robustmnist_case as case
ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=Path('examples/robustmnist'));ap.add_argument('--cache-dir',type=Path,default=Path('/tmp/robustmnist-case'));args=ap.parse_args()
result=json.loads((args.output/'reproduction.json').read_text())
labels=['clean']+list(result['transforms'])
values=[result['clean']['accuracy']]+[v['accuracy'] for v in result['transforms'].values()]
fig,ax=plt.subplots(figsize=(9,4));ax.bar(labels,[v*100 for v in values]);ax.set(ylabel='Accuracy (%)',ylim=(95,100),title='RobustMNIST: 2,000 diagnostic handwritten digits');ax.tick_params(axis='x',rotation=55);fig.tight_layout();fig.savefig(args.output/'accuracy.png',dpi=150);plt.close(fig)
torch.set_num_threads(2)
spec=importlib.util.spec_from_file_location('architecture',args.cache_dir/'model.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
model=module.HierarchicalNetwork().eval();model.load_state_dict(load_file(args.cache_dir/'model.safetensors'))
x,y=case.dataset(args.cache_dir);ix=case.split_ids()['diagnostic'];x=x[ix];y=y[ix];rot=case.transform(x,'rotation:30')
a=case.predict(model,x).argmax(-1);b=case.predict(model,rot).argmax(-1)
selected=((a==y)&(b!=y)).nonzero().flatten()[:6]
fig,axes=plt.subplots(2,len(selected),figsize=(10,4))
for j,i in enumerate(selected):
    for row,images in enumerate([x,rot]):
        axes[row,j].imshow(images[i,0],cmap='gray',vmin=0,vmax=1);axes[row,j].axis('off')
    axes[0,j].set_title(f'True {int(y[i])}; pred {int(a[i])}')
    axes[1,j].set_title(f'+30°; pred {int(b[i])}')
fig.suptitle('First six clean-correct → rotated-wrong diagnostic examples');fig.tight_layout();fig.savefig(args.output/'failures.png',dpi=150)
case.save(args.output,'figure-provenance.json',{'source_ids':ix[selected].tolist(),'model_forward_batches':64,'input_examples':4000,'selection':'first six diagnostic errors at +30 degrees; no sealed examples'})
