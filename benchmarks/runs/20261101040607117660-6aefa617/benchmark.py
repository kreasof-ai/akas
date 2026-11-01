"""Compare shared native expert tiles at serving shapes, retaining exact outputs."""
from pathlib import Path
from contextlib import ExitStack
import ctypes as c,json
import numpy as np,tensor
from akas.benchmarking.history import Recorder
from akas.compilation.selection import source
from akas.kernels.definition import define
from akas.kernels.contracts import sample_arguments
from akas.ops.moe.validation import tile_tables
from tensor.compiler.frontend.signature import frontend_arguments
from tensor.compiler.tuning import measure,cuda_working_set
from tensor.runtime.dtypes import decode_bfloat16,encode_bfloat16

main=Path('/home/sagemaker-user/akas');work=main/'build/worktrees/expert-pipeline'
root=work/'build/lfm-serving/extended-expert-tiles';root.mkdir(parents=True,exist_ok=True)
metadata=dict(runtime='akas',purpose='expert-tile-benchmark',provider='cuda',qualification_scope='Serving-sized expert operations with bitwise schedule comparison and separately retained independent operation tests; not model throughput')
rec=Recorder(main/'benchmarks',metadata,source_root=work,kind='benchmark',change='Measure larger expert row tiles through the existing shared route-map contract',dependency_sources={'tensor':main.parent/'tensor'})
rec.artifact(Path(__file__),'benchmark.py')
rec.artifact(work/'build/lfm-serving/extended-map-gpu-tests.log','independent-operation-tests.log')
print('STARTED',rec.id,flush=True)
rows=[];entries=[]
try:
 with tensor.Device() as device,ExitStack() as lifetime:
  prepare,conditioning=lifetime.enter_context(cuda_working_set(device,256<<20,out=root/'working-set',target='sm_89'))
  for count in (2048,8192):
   for kind in ('fp8_expert_swiglu','expert_partial'):
    paired=kind=='fp8_expert_swiglu';columns,depth=(1536,2048) if paired else (2048,1536)
    base=dict(r=count,rows=count,top=4,experts=64,o=columns,k=depth,routed_input=not paired,partitions=1,block_m=32,compact=True,tile_map=True,max_block_m=128)
    if paired:base.update(product_dtype='bfloat16',activation_dtype='bfloat16')
    else:base['output_dtype']='bfloat16'
    schedules=([dict(block_m=32,threads=256,stages=1,pair_layout='wide'),dict(block_m=64,threads=256,stages=2,pair_layout='wide'),dict(block_m=64,threads=256,stages=1,pair_layout='separate'),dict(block_m=128,threads=512,stages=1,pair_layout='separate'),dict(block_m=128,threads=512,stages=2,pair_layout='separate')]
      if paired else [dict(block_m=64,columns=128,threads=256,stages=1),dict(block_m=128,columns=128,threads=512,stages=1),dict(block_m=128,columns=128,threads=512,stages=2),dict(block_m=128,columns=64,threads=256,stages=1),dict(block_m=128,columns=64,threads=256,stages=2)])
    signature=frontend_arguments(define(dict(kind=kind,parameters=base,contract='reference')), {})
    initial=sample_arguments(signature);rng=np.random.default_rng(1917)
    for name in ('x','w','w2'):
     if name not in initial:continue
     shape=initial[name].shape
     bits=rng.integers(32,81,shape,dtype='uint8')
     initial[name]=bits | (rng.integers(0,2,shape,dtype='uint8')*128)
    for name in ('activation_scales','scales','scales2'):
     if name not in initial:continue
     values=rng.uniform(.01,.02,initial[name].shape).astype('float32')
     initial[name]=decode_bfloat16(encode_bfloat16(values)) if name!='activation_scales' else values
    initial['counts'][:]=0;initial['routes'][:]=-1
    for row in range(count):
     for rank in range(4):
      expert=(row*17+rank*11)%64;index=initial['counts'][expert]
      initial['routes'][expert,index]=row*4+rank;initial['counts'][expert]+=1
    initial['tile_experts'],initial['tile_offsets']=tile_tables(initial['counts'],initial['tile_experts'].shape[1],6)
    outputs=[a['name'] for a in signature if a['name'] in ('out','output_scales')]
    for name in outputs:initial[name][:]=0
    with ExitStack() as case:
     buffers={a['name']:device.from_numpy(initial[a['name']],dtype=a['dtype']) for a in signature}
     for value in buffers.values():case.callback(value.release)
     del initial
     baseline=None;results=[]
     for index,schedule in enumerate(schedules):
      parameters={**base,**schedule,'activation_vector':16,'mma_dtype':'float8_e4m3fn'}
      label=f'{count}-{kind}-{index}';entry=root/(label+'.py');artifact=entry.with_suffix('.tbin')
      entry.write_text(source(kind,parameters,contract='reference',target='sm_89').render())
      if not artifact.exists():tensor.build(entry,artifact,target='sm_89')
      rec.artifact(entry,label+'.py');rec.artifact(artifact,label+'.tbin')
      with device.load(artifact) as executable:
       bound=executable.bind(*(buffers[a['name']] for a in executable.manifest['arguments']));bound.launch()
       actual={name:buffers[name].to_numpy() for name in outputs}
       if baseline is None:baseline=actual
       else:
        for name in outputs:np.testing.assert_array_equal(actual[name],baseline[name])
       attributes={};device.driver.lib.cuFuncGetAttribute.argtypes=[c.POINTER(c.c_int),c.c_int,c.c_void_p]
       for name,attribute in (('registers_per_thread',4),('local_bytes_per_thread',3)):
        value=c.c_int();device.driver.call('cuFuncGetAttribute',c.byref(value),attribute,executable.function);attributes[name]=value.value
       timing=measure(device,bound.launch,warmup=2,samples=7,repeats=3,reset=prepare)
       row=dict(rows=count,kind=kind,schedule=schedule,bitwise=True,resources=attributes,launch=executable.manifest['launch'],timing=timing)
       rows.append(row);results.append(row);print('RESULT',count,kind,schedule,attributes,timing['median_seconds'],flush=True)
     best=min(results,key=lambda row:row['timing']['median_seconds'])
     implementation='moe.cuda.fp8-swiglu:mapped' if paired else 'moe.cuda.fp8-bucketed:mapped'
     entries.append(dict(operation=kind,parameters=base,schedule={**best['schedule'],'activation_vector':16,'mma_dtype':'float8_e4m3fn','implementation':implementation}))
 report=dict(results=rows,cache_conditioning=conditioning)
 for name,data in [('results.json',report),('profile.json',dict(schema='tensor.schedule-profile.v1',provider='cuda',target='sm_89',entries=entries))]:
  path=root/name;path.write_text(json.dumps(data,indent=2)+'\n');rec.artifact(path,name)
 rec.update(dict(metadata=metadata,results=[]),final=True,status='passed');print('RUN',rec.id,flush=True)
except BaseException as error:
 if rec.record['completed_at'] is None:rec.update(dict(metadata=metadata,results=[],error=repr(error)),final=True,status='failed')
 raise
