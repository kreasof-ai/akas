"""Contraction MFU and logical weight/KV MBU bounds for the retained AR run.

These are model-level accounting bounds, not hardware-counter measurements.
The weights are charged once per phase; unnecessary traffic is not a ceiling.
"""
from pathlib import Path
from math import prod
import argparse,json,statistics
from akas.artifacts.schema import identity
from akas.benchmarking.history import Recorder,digest
from akas.compilation.transforms.projection import projection_algorithm
from akas.ops.linear.cuda import MINIMUM_MMA_GROUPS,MAXIMUM_GEMV_ROWS
from akas.models.lfm2 import LFM2MoE

parser=argparse.ArgumentParser()
parser.add_argument('--run',required=True)
parser.add_argument('--program',type=Path,default=Path('build/lfm-serving/program-concurrent-fused'))
parser.add_argument('--profile',type=Path)
args=parser.parse_args()
root=Path('build/lfm-serving')
run_path=Path('benchmarks/runs')/args.run/'run.json'
run=json.loads(run_path.read_text())
assert run['completed_at'] is not None and run['status']=='passed'
assert run['metadata']['kv_bits']==16 and run['metadata']['mode']=='ar'
assert digest(args.program/'program.json')==run['metadata']['program_sha256']
program=json.loads((args.program/'program.json').read_text())
config=LFM2MoE.checkpoint_type(root.parent/'checkpoints/lfm2.5-24b-a2b-fp8',tokenizer=False).config
bytes_per_dtype={'float32':4,'float16':2,'bfloat16':2,'uint8':1,'int32':4}
theoretical={'fp8_tensor':733e12,'fp16_tensor':366.5e12,'fp32_simt':91.6e12}
calibration=json.loads((root/'device-ceilings/extended-results.json').read_text())
read_peak=next(r['gigabytes_per_second']*1e9 for r in calibration['calibration'] if r['kind']=='streaming_read_reduction')

def phase_accounting(phase, observed_counts=None):
 data=json.loads((args.program/phase['path']).read_text())
 graph=data['forward'];buffers={b['name']:b for b in graph['buffers']}
 flops={key:0 for key in theoretical};weights={};unclassified=set()
 def base(name):
  while buffers[name].get('alias_of'):name=buffers[name]['alias_of']
  return name
 for call in graph['calls']:
  p=call['parameters'];kind=call['kind']
  kernel=data['kernels'][identity(kind,p)]
  schedule=kernel['schedule'];implementation=schedule['implementation']
  if kind in ('linear','matmul','fp8_linear','projection_partial','expert_partial','expert_swiglu','fp8_expert_swiglu'):
   amount=2*p['r']*p['k']*p['o']
   if kind in ('expert_partial','expert_swiglu','fp8_expert_swiglu'):amount*=p['top']
   if kind in ('expert_swiglu','fp8_expert_swiglu'):amount*=2
   if kind=='matmul':precision='fp32_simt' if p.get('dtype','float32')=='float32' else 'fp16_tensor'
   elif kind=='linear':
    if implementation=='linear.cuda.floating':
     options={**p,**schedule,'maximum_gemv_rows':schedule.get('maximum_gemv_rows',MAXIMUM_GEMV_ROWS)}
     algorithm=projection_algorithm(options,minimum_groups=schedule.get('minimum_mma_groups',MINIMUM_MMA_GROUPS),default_tile=(32,64,32))
     precision='fp32_simt' if algorithm=='gemv' else 'fp16_tensor'
    else:
     assert implementation=='linear.cuda.floating_warp',implementation
     precision='fp32_simt'
   elif 'fp8-gemv' in implementation:precision='fp32_simt'
   else:precision='fp16_tensor' if {**p,**schedule}.get('mma_dtype')=='float16' else 'fp8_tensor'
   flops[precision]+=amount
  else:unclassified.add(kind)
  if kind=='embedding':continue # The tied full output head is charged below.
  names=(*call['arguments'],*call['bindings'].values())
  for name in names:
   if buffers[name]['role']!='weight':continue
   name=base(name);spec=buffers[name]
   size=prod(spec['shape'])*bytes_per_dtype[spec['dtype']]
   lower=upper=size
   if kind in ('expert_partial','expert_swiglu','fp8_expert_swiglu'):
    assert spec['shape'][0]==p['experts']
    if observed_counts is not None:
     counts=observed_counts[call['bindings']['counts']]
     assert len(counts)==p['experts'] and sum(counts)==p['r']*p['top']
     lower=upper=size*sum(c>0 for c in counts)//p['experts']
    else:
     lower=size*p['top']//p['experts']
     upper=size*min(p['experts'],p['r']*p['top'])//p['experts']
   previous=weights.get(name,(0,0))
   weights[name]=(max(previous[0],lower),max(previous[1],upper))
 return dict(flops_by_precision=flops,
   ideal_contraction_seconds=sum(flops[k]/v for k,v in theoretical.items()),
   weight_payload_bytes_bounds=[sum(v[i] for v in weights.values()) for i in (0,1)],
   excluded_from_contraction_flops=sorted(unclassified))

rows=[]
for row in run['results']:
 batch=row['case']['batch'];prefix=row['case']['prompt'];outputs=row['case']['generated']
 phase=next(p for p in program['phases'] if p['phase']=='decode' and p['rows']==batch)
 cost=phase_accounting(phase)
 seconds=row['summary']['inter_token_seconds']['median']
 average_kv_length=prefix+outputs/2
 kv_read=2*config.layers.count('attention')*batch*average_kv_length*config.kv_heads*config.head_dim*2
 kv_write=2*config.layers.count('attention')*batch*config.kv_heads*config.head_dim*2
 bounds=[b+kv_read+kv_write for b in cost['weight_payload_bytes_bounds']]
 rows.append(dict(batch=batch,scope='average streamed decode interval',seconds_per_step=seconds,**cost,
   contraction_mfu_lower_bound=cost['ideal_contraction_seconds']/seconds,
   kv_cache_read_bytes_per_step=kv_read,kv_cache_write_bytes_per_step=kv_write,
   logical_weight_kv_mbu_bounds_vs_nominal=[b/seconds/864e9 for b in bounds],
   logical_weight_kv_mbu_bounds_vs_measured_read=[b/seconds/read_peak for b in bounds]))
observed=[]
if args.profile:
 profile=json.loads(args.profile.read_text())
 assert profile['program_sha256']==run['metadata']['program_sha256']
 for case in profile['cases']:
  batch,chunk,position=(case[k] for k in ('batch','chunk','position'))
  phase=next(p for p in program['phases'] if (p['phase'],p['rows'],p['chunk'])==
             ('decode' if chunk==1 else 'prefill',batch,chunk))
  counts={o['outputs']['counts']['buffer']:o['outputs']['counts']['values'] for o in case['observations']}
  cost=phase_accounting(phase,counts)
  assert cost['weight_payload_bytes_bounds'][0]==cost['weight_payload_bytes_bounds'][1]
  seconds=statistics.median(case['timing']['graph_span_seconds'])
  kv_unit=2*config.layers.count('attention')*batch*config.kv_heads*config.head_dim*2
  kv_read=kv_unit*(position+chunk);kv_write=kv_unit*chunk
  logical=cost['weight_payload_bytes_bounds'][0]+kv_read+kv_write
  observed.append(dict(batch=batch,chunk=chunk,position=position,
   scope='captured GPU phase with routing observed in a separate untimed deterministic replay',
   gpu_seconds=seconds,**cost,kv_read_bytes=kv_read,kv_write_bytes=kv_write,
   contraction_mfu_lower_bound=cost['ideal_contraction_seconds']/seconds,
   logical_weight_kv_mbu_vs_measured_read=logical/seconds/read_peak,
   active_experts=[sum(c>0 for c in values) for values in counts.values()]))
metadata=dict(runtime='akas',purpose='utilization-accounting',provider='cuda',native_run=args.run,
 model=run['metadata']['model'],program_sha256=run['metadata']['program_sha256'],
 qualification_scope='Contraction MFU lower bounds and logical weight/KV MBU bounds; no hardware saturation claim',
 theoretical_dense_peaks_flops_per_second=theoretical,nominal_dram_bytes_per_second=864e9,
 measured_read_bytes_per_second=read_peak,hardware_counters='denied ERR_NVGPUCTRPERM',
 source_specification='https://www.nvidia.com/en-sg/data-center/l40s/')
report=dict(metadata=metadata,rows=rows,observed_phases=observed,assumptions=[
 'FMA counts as two FLOPs; attention and pointwise FLOPs are excluded explicitly.',
 'MBU counts each selected weight payload once and the minimally shared 16-bit KV reads/writes.',
 'Expert weight bounds span top-k through min(experts, rows*top-k) distinct experts; C1 bounds coincide.',
 'This accounting excludes unnecessary traffic and does not measure DRAM counters or prove a ceiling.',
 'Timing includes scheduler and streamed delivery, using the retained mean decode interval per token.'])
path=root/'hardware-accounting.json';path.write_text(json.dumps(report,indent=2)+'\n')
record=Recorder('benchmarks',metadata,kind='benchmark',change='Account for MFU and logical MBU without asserting hardware saturation')
for source,name in ((Path(__file__),'hardware-accounting.py'),(path,'hardware-accounting.json'),
                    (run_path,'native-run.json'),(root/'device-ceilings/extended-results.json','device-calibration.json')):
 record.artifact(source,name)
if args.profile:record.artifact(args.profile,'captured-profile.json')
record.update(dict(metadata=metadata,results=[]),final=True,status='passed')
print('RUN',record.id)
for row in rows:print(row['batch'],row['contraction_mfu_lower_bound'],row['logical_weight_kv_mbu_bounds_vs_measured_read'])
for row in observed:print('OBSERVED',row['batch'],row['chunk'],row['contraction_mfu_lower_bound'],row['logical_weight_kv_mbu_vs_measured_read'])
