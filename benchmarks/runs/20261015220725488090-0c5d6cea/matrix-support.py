import json
from pathlib import Path
from tools.campaign import entries
from akas.optimization.context import symbol
from akas.benchmarking.history import write,source_identity,utc
root=Path.cwd();spec=json.loads(Path('benchmarks/campaigns/standard-v1.json').read_text());loaded={};rows=[]
for entry in entries(spec,root):
 c=entry['context'];model=symbol(c['definition'])
 if c['checkpoint'] not in loaded:loaded[c['checkpoint']]=model.checkpoint_type(c['checkpoint'])
 checkpoint=loaded[c['checkpoint']];options=dict(model.model_defaults)
 if c['kv_dtype']:options['kv_dtype']=c['kv_dtype']
 row=dict(id=entry['id'],context=c,effective_model_options=options)
 try:
  args=dict(rows=1,chunk=32,max_requests=256,max_context=2176+(2 if c['mode']=='mtp' else 0),provider=c['backend'],**options)
  model.compose(checkpoint,**args)
  if c['mode']=='mtp':
   if not model.speculative_head:raise ValueError('model declares no speculative head')
   model.compose(checkpoint,**{**args,'chunk':2,'phase':'verify'})
   model.compose(checkpoint,**{**args,'phase':'proposal'})
  row['status']='composition_supported'
 except (ValueError,TypeError,NotImplementedError,AttributeError) as error:
  row.update(status='unsupported',reason=str(error))
 rows.append(row)
write(Path('build/optimization-pilots/matrix-support.json'),dict(schema='akas.matrix-support.v1',source=source_identity(root),created_at=utc(),scope='Checkpoint loading and model composition; shader/runtime qualification is performed by the bounded optimizer',rows=rows))
print('supported',sum(r['status']=='composition_supported' for r in rows),'unsupported',sum(r['status']=='unsupported' for r in rows))
