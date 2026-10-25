import json
from pathlib import Path
p=Path('build/vulkan-next')
data=json.loads((p/'mixed-profile.json').read_text())
for e in data['entries']:
    kind=e['operation']
    if kind.startswith('attention'):
        e['parameters']['cache_padding']=128
    if kind in ('attention_adaptive_partial','attention_merge'):
        e['parameters']['splits']=64
(p/'padded-profile.json').write_text(json.dumps(data,indent=2))
(p/'policy-padded.json').write_text(json.dumps(dict(attention=dict(splits=64,adaptive_splits=True),linear=dict(stage_weights=True))))
