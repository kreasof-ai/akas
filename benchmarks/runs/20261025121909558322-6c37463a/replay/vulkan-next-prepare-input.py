import json
from pathlib import Path
p=Path('build/vulkan-next');d=json.loads((p/'mixed-profile.json').read_text())
for e in d['entries']:
 if e['operation'] in ('attention_adaptive_partial','attention_merge'):e['parameters']['splits']=64
 if e['parameters'].get('half_accum'):e['parameters']['input_dtype']='float16'
(p/'mixed-input-profile.json').write_text(json.dumps(d,indent=2))
(p/'policy-mixed-input.json').write_text(json.dumps(dict(attention=dict(splits=64,adaptive_splits=True),linear=dict(stage_weights=True,stage_inputs=True))))
