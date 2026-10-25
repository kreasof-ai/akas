import json
from pathlib import Path
p=Path('build/vulkan-next')
data=json.loads((p/'padded-profile.json').read_text())
for e in data['entries']:
 if e['operation']=='attention_adaptive_partial':e['schedule']['key_rows']=64
(p/'padded64-profile.json').write_text(json.dumps(data,indent=2))
