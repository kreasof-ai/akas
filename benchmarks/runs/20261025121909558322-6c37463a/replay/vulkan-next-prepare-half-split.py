import json,copy
from pathlib import Path
p=Path('build/vulkan-next')
data=json.loads((p/'mixed-profile.json').read_text())
extra=[]
for e in data['entries']:
 k,v=e['operation'],e['parameters']
 if k in ('attention_adaptive_partial','attention_merge'):v['splits']=64
 if k in ('linear','ffn','linear_add') and v.get('r',1)>1 and v.get('type') in (1,2):
  v.pop('q16',None);v.pop('integer',None);v['half_accum']=True
  v['type']=1 if v['r']>=128 else 2
  layout=(2 if v['o']>=6144 or v['k']==10752 else 1) if v['r']>=128 else (0 if k=='ffn' else 3 if v['o']==6144 else 0)
  e['schedule']=dict(implementation='linear.webgpu.short_half',half_layout=layout,half_k=32 if v['r']>=128 else 16,half_native=True,half_pad=1)
  if k=='ffn' and v['r']>=128:
   e['operation']='linear'
data['entries']+=extra
(p/'half-split-profile.json').write_text(json.dumps(data,indent=2))
(p/'policy-half-split.json').write_text(json.dumps(dict(attention=dict(splits=64,adaptive_splits=True),linear=dict(stage_weights=True,split_ffn=True,split_ffn_min_rows=128))))
