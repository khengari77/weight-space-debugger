import ast
from pathlib import Path
import torch
SOURCE=Path(__file__).resolve().parents[1]/'examples/whisper/run.py'
def test_acceptance_rejects_silencing_real_speech():
 tree=ast.parse(SOURCE.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='acceptance')
 scope={};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(SOURCE),'exec'),scope)
 f=scope['acceptance'];base={'hallucination':.8,'speech_wer':{'wer':5}}
 patch={'hallucination':.1,'speech_kept':1.,'speech_wer':{'wer':5}}
 assert f(base,patch)
 assert not f(base,{**patch,'speech_kept':.98})
 assert not f(base,{**patch,'hallucination':.5})
 assert not f(base,{**patch,'speech_wer':{'wer':6}})
def test_sparse_row_column_restoration_is_exact():
 layer=torch.nn.Sequential(torch.nn.Linear(5,7),torch.nn.GELU(),torch.nn.Linear(7,5))
 before={k:v.clone() for k,v in layer.state_dict().items()}
 with torch.no_grad():
  layer[0].weight[2].fill_(.2);layer[0].bias[2]=-.1;layer[2].weight[:,2].fill_(.3)
 assert sum(int((v!=before[k]).sum()) for k,v in layer.state_dict().items())==11
 layer.load_state_dict(before)
 assert all(torch.equal(v,before[k]) for k,v in layer.state_dict().items())
