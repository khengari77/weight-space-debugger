"""Replace cross-split environmental source overlap before any inference."""
import csv,hashlib,io,json,urllib.request
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
root=Path('/tmp/whisper-study')
rows=json.loads((root/'manifest.json').read_text())
practice={r['group'] for r in rows if r['split']!='test'}
bad=[r for r in rows if r['split']=='test' and r['group'] in practice]
rows=[r for r in rows if r not in bad]
url='https://raw.githubusercontent.com/karolpiczak/ESC-50/master/'
metadata=list(csv.DictReader(io.StringIO(urllib.request.urlopen(url+'meta/esc50.csv').read().decode())))
used={r['id'] for r in rows}
for old in bad:
 item=next(r for r in metadata if int(r['fold'])>=3 and int(r['target']) not in range(20,30) and r['filename'] not in used and 'esc:'+r['src_file'] not in practice)
 a,sr=sf.read(io.BytesIO(urllib.request.urlopen(url+'audio/'+item['filename']).read()))
 name=item['filename']+'.npy';np.save(root/'data'/name,resample_poly(a,16000,sr).astype('float32'))
 rows.append(dict(id=item['filename'],group='esc:'+item['src_file'],split='test',kind='nonspeech',audio=name,ref='',category=item['category'],sha256=hashlib.sha256((root/'data'/name).read_bytes()).hexdigest()))
 used.add(item['filename'])
(root/'manifest.json').write_text(json.dumps(rows,indent=2))
(root/'split_correction.json').write_text(json.dumps({'removed':bad,'reason':'Original-source overlap found before any model evaluation; replaced with unused test-fold clips from separate sources.'},indent=2))
for a,b in [('diagnostic','development'),('diagnostic','test'),('development','test')]:
 assert not({r['group'] for r in rows if r['split']==a}&{r['group'] for r in rows if r['split']==b})
print('Source groups verified:',len(rows))
