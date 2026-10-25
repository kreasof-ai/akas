import json
from pathlib import Path
p=Path('build/vulkan-next');d=json.loads((p/'half-split-profile.json').read_text())
for e in d['entries']:
 if e['operation']=='attention_adaptive_partial':e['schedule']['key_rows']=16
(p/'half16-profile.json').write_text(json.dumps(d,indent=2))
