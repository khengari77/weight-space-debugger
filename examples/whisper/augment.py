import json,hashlib
from pathlib import Path
import numpy as np
root=Path('/tmp/whisper-study');records=json.loads((root/'manifest.json').read_text())
for split in ['diagnostic','development','test']:
 speech=[r for r in records if r['split']==split and r['kind']=='speech'];ns=[r for r in records if r['split']==split and r['kind']=='nonspeech']
 for i,r in enumerate(speech):
  a=np.load(root/'data'/r['audio']);r['augmentation']='clean'
  if i%4==1 and len(a)<25*16000:a=np.pad(a,(16000,16000));r['augmentation']='padding'
  if i%4==2:a=a*.1;r['augmentation']='quiet'
  if i%4==3:
   other=ns[i%len(ns)];b=np.resize(np.load(root/'data'/other['audio']),len(a));b=b/(np.sqrt((b*b).mean())+1e-8)*np.sqrt((a*a).mean())*.316;a=np.clip(a+b,-1,1);r['augmentation']='background_10dB';r['background_source']=other['id']
  np.save(root/'data'/r['audio'],a.astype('float32'))
for r in records:
 r['group']=('esc:' if r['kind']=='nonspeech' else 'speaker:')+r['group'];r['sha256']=hashlib.sha256((root/'data'/r['audio']).read_bytes()).hexdigest()
(root/'manifest.json').write_text(json.dumps(records,indent=2))
