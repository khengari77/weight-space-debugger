import urllib.request,tarfile,io,csv,json,random,hashlib
from pathlib import Path
import soundfile as sf
import numpy as np
from scipy.signal import resample_poly
root=Path('/tmp/whisper-study');data=root/'data';data.mkdir(exist_ok=True)
def download(url,path):
 if not path.exists():
  temp=path.with_suffix(path.suffix+'.partial')
  urllib.request.urlretrieve(url,temp);temp.replace(path)
rng=random.Random(2026);records=[]
meta=urllib.request.urlopen('https://raw.githubusercontent.com/karolpiczak/ESC-50/master/meta/esc50.csv').read().decode()
rows=list(csv.DictReader(io.StringIO(meta)))
rows=[r for r in rows if int(r['target']) not in range(20,30)]
rng.shuffle(rows)
for split,folds,n in [('diagnostic',[1],40),('development',[2],40),('test',[3,4,5],200)]:
 selected=[r for r in rows if int(r['fold']) in folds][:n]
 for r in selected:
  p=data/r['filename'];download('https://raw.githubusercontent.com/karolpiczak/ESC-50/master/audio/'+r['filename'],p)
  a,sr=sf.read(p);a=resample_poly(a,16000,sr).astype('float32');name=r['filename']+'.npy';np.save(data/name,a)
  records.append(dict(id=r['filename'],group=r['src_file'],split=split,kind='nonspeech',audio=name,ref='',category=r['category']))
 print(split,'environmental ready',flush=True)
for name,n in [('dev-clean',80),('test-clean',200)]:
 p=root/(name+'.tar.gz');download('https://www.openslr.org/resources/12/'+name+'.tar.gz',p)
 with tarfile.open(p) as tar:
  texts={}
  for m in tar.getmembers():
   if m.name.endswith('.trans.txt'):
    for line in tar.extractfile(m).read().decode().splitlines():
     k,v=line.split(' ',1);texts[k]=v
  files=[m for m in tar.getmembers() if m.name.endswith('.flac')];speakers=sorted({m.name.split('/')[-3] for m in files});rng.shuffle(speakers)
  assignments=[('test',speakers,200)] if name=='test-clean' else [('diagnostic',speakers[:len(speakers)//2],40),('development',speakers[len(speakers)//2:],40)]
  for split,sp,count in assignments:
   pools={s:[m for m in files if m.name.split('/')[-3]==s] for s in sp};chosen=[]
   while len(chosen)<count:
    for s in sp:
     if pools[s] and len(chosen)<count:chosen.append(pools[s].pop(0))
   for m in chosen:
    a,sr=sf.read(io.BytesIO(tar.extractfile(m).read()));k=Path(m.name).stem;fname=k+'.npy';np.save(data/fname,a.astype('float32'))
    records.append(dict(id=k,group=k.split('-')[0],split=split,kind='speech',audio=fname,ref=texts[k],category='speech'))
 print(name,'ready',flush=True)
for r in records:r['sha256']=hashlib.sha256((data/r['audio']).read_bytes()).hexdigest()
(root/'manifest.json').write_text(json.dumps(records,indent=2));print('COMPLETE',len(records),flush=True)
