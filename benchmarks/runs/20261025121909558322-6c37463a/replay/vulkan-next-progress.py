import json
from pathlib import Path
for path in Path('build/vulkan-next/tune-half-deeper/search-cache').glob('*.json'):
 d=json.loads(path.read_text());print(d.keys());print(str(d)[:1000]);print(str(d)[-2300:])
