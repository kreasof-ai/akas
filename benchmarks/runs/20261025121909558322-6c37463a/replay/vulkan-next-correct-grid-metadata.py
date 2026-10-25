import ast,json
from pathlib import Path
root=Path('build/vulkan-next/decode-grid-v3');raw=json.loads((root/'results.json').read_text());rows=[]
for row in raw:
 n=row['granule'];source=(root/f"r{row['key_rows']}-g{n}"/'kernel.py').read_text();tree=ast.parse(source)
 call=next(n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='define')
 params=ast.literal_eval(call.args[0])['parameters'];assert params['splits']==n and params['granule']==128
 rows.append({**row,'splits':n,'granule':params['granule']})
(root/'results-corrected.json').write_text(json.dumps(dict(note='Correct axis labels recovered from emitted kernel parameters; original results.json is retained unchanged.',rows=rows),indent=2))
