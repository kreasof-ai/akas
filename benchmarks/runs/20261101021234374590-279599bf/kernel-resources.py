"""Read CUDA function resources and theoretical residency without performance counters."""
from pathlib import Path
import ctypes as c,json,math
import tensor
from akas.benchmarking.history import Recorder,digest
main=Path('/home/sagemaker-user/akas');base=Path('build/lfm-serving')
metadata=dict(runtime='akas',purpose='kernel-resource-attribution',provider='cuda',qualification_scope='CUDA function resources and theoretical block residency, not measured utilization or a saturation claim')
rec=Recorder(main/'benchmarks',metadata,kind='diagnostic',change='Locate expert occupancy limits in current and staged K variants',dependency_sources={'tensor':Path('/home/sagemaker-user/tensor')})
rec.artifact(Path(__file__),'kernel-resources.py');print('STARTED',rec.id,flush=True)
paths=[]
r=main/'build/lfm-serving/program-concurrent-fused';m=json.loads((r/'program.json').read_text())
for phase in m['phases']:
 if phase['phase']!='prefill' or phase['chunk']!=2048:continue
 d=json.loads((r/phase['path']).read_text());seen=set()
 for k in d['kernels'].values():
  if k['kind'] in ('fp8_expert_swiglu','expert_partial') and k['kind'] not in seen:
   seen.add(k['kind']);paths.append((f"serving-b{phase['rows']}-{k['kind']}",r/k['path']))
for name in ('preloaded-prefill-experts','k64-prefill-experts'):
 for p in (base/name/'search-kernels').glob('*.tbin'):paths.append((name,p))
rows=[]
try:
 with tensor.Device() as device:
  driver=device.driver
  driver.lib.cuFuncGetAttribute.argtypes=[c.POINTER(c.c_int),c.c_int,c.c_void_p]
  driver.lib.cuOccupancyMaxActiveBlocksPerMultiprocessor.argtypes=[c.POINTER(c.c_int),c.c_void_p,c.c_int,c.c_size_t]
  seen=set()
  for label,path in paths:
   sha=digest(path)
   if (label,sha) in seen:continue
   seen.add((label,sha))
   with device.load(path) as executable:
    values={}
    for name,attribute in (('registers_per_thread',4),('local_bytes_per_thread',3),('static_shared_bytes',1)):
     value=c.c_int();driver.call('cuFuncGetAttribute',c.byref(value),attribute,executable.function);values[name]=value.value
    launch=executable.manifest['launch'];shared=launch['shared_memory_bytes'];threads=math.prod(launch['block'])
    if shared>49152:driver.call('cuFuncSetAttribute',executable.function,8,shared)
    blocks=c.c_int();driver.call('cuOccupancyMaxActiveBlocksPerMultiprocessor',c.byref(blocks),executable.function,threads,c.c_size_t(shared))
    row=dict(label=label,artifact_sha256=sha,**values,threads=threads,dynamic_shared_bytes=shared,active_blocks_per_sm=blocks.value,active_warps_per_sm=blocks.value*threads//32,arguments=executable.manifest['arguments'])
    rows.append(row);print({k:v for k,v in row.items() if k!='arguments'},flush=True)
 out=base/'kernel-resources.json';out.write_text(json.dumps(dict(rows=rows),indent=2)+'\n');rec.artifact(out,'kernel-resources.json')
 rec.update(dict(metadata=metadata,results=[]),final=True,status='passed');print('RUN',rec.id,flush=True)
except BaseException as error:
 if rec.record['completed_at'] is None:rec.update(dict(metadata=metadata,results=[],error=repr(error)),final=True,status='failed')
 raise
