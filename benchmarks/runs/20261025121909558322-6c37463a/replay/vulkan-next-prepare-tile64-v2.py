import json
from pathlib import Path
p=Path('build/vulkan-next');d=json.loads((p/'mixed-input-profile.json').read_text())
for e in d['entries']:
 if e['operation']=='attention_adaptive_partial':e['schedule']['key_rows']=64
(p/'mixed-tile64-profile.json').write_text(json.dumps(d,indent=2))
