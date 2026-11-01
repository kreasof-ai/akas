"""AOT preparation and complete expert-chain quantizer placement measurements."""
from pathlib import Path
from contextlib import ExitStack
import argparse,json,zipfile
import numpy as np,tensor
from akas.graph import Graph
from akas.compilation.policy import lower
from akas.compilation.compile import produce
from akas.execution.forward import Forward
from akas.benchmarking.history import Recorder
from akas.models.lfm2 import LFM2MoE
from akas.checkpoints.upload import upload_shards
from akas.ops.moe.validation import tile_tables
from akas.quantization._shared.e4m3 import encode_storage
from tensor.compiler.tuning import cuda_working_set,measure

parser=argparse.ArgumentParser();parser.add_argument('--compile-only',action='store_true');args=parser.parse_args()
main=Path('/home/sagemaker-user/akas');work=Path.cwd();root=work/'build/lfm-serving/staged-prefill-choice';root.mkdir(parents=True,exist_ok=True)
# Fused quantization requires N128. Separate quantization retains K128 groups
# while allowing smaller paired-projection column tiles.
candidates=[(True,64,128,256,2,'wide'),(True,128,128,512,2,'separate')]
candidates += [(False,m,64,t,st,l) for m,t,st,l in ((32,128,1,'wide'),(32,128,2,'wide'),(64,256,1,'wide'),(64,256,2,'wide'),(64,256,2,'separate'),(64,128,1,'separate'))]

candidates += [('scaled',m,128,128,1,l) for m,l in ((16,'wide'),(32,'wide'),(32,'separate'))]

def compose(rows,candidate):
 fused,m,n,t,s,l=candidate;graph=Graph(contract='reference')
 shapes=dict(x=((rows,2048),'uint8'),activation_scales=((rows,16),'float32'),
   w=((64,1536,2048),'uint8'),w2=((64,1536,2048),'uint8'),down=((64,2048,1536),'uint8'),
   w_scales=((64,12,16),'bfloat16'),w2_scales=((64,12,16),'bfloat16'),down_scales=((64,16,12),'bfloat16'),
   counts=((64,),'int32'),routes=((64,rows),'int32'),tile_experts=((6,rows+64),'int32'),tile_offsets=((6,rows+64),'int32'),out=((rows,4,2048),'bfloat16'))
 for name,(shape,dtype) in shapes.items():graph.buffer(name,shape,dtype,role='output' if name=='out' else 'input')
 encoded=graph.buffer('activated',(rows,4,1536),'uint8');scales=graph.buffer('activated_scales',(rows,4,12),'float32')
 activated=encoded if fused else graph.buffer('activated_float',(rows,4,1536),'bfloat16')
 shared={name:graph.buffers[name] for name in ('counts','routes','tile_experts','tile_offsets')}
 common=dict(r=rows,rows=rows,top=4,experts=64,partitions=1,compact=True,tile_map=True,max_block_m=128,activation_vector=16,mma_dtype='float8_e4m3fn' if fused is True else 'float16',**({} if fused is True else dict(dequantize_dtype='float16')))
 graph('fp8_expert_swiglu' if fused else 'expert_swiglu',dict(common,k=2048,o=1536,block_m=m,columns=n,threads=t,stages=s,pair_layout=l,product_dtype='bfloat16',activation_dtype='bfloat16',**({} if fused else dict(weight_format='fp8',input_dtype='uint8',output_dtype='bfloat16'))),
   **{name:graph.buffers[name] for name in ('x','activation_scales','w','w2')},scales=graph.buffers['w_scales'],scales2=graph.buffers['w2_scales'],**shared,out=activated,**(dict(output_scales=scales) if fused else {}))
 if not fused:graph('quantize',dict(r=rows,top=4,k=1536,input_dtype='bfloat16'),activated,encoded,scales)
 graph('expert_partial',dict(common,k=1536,o=2048,block_m=64,columns=128,threads=256,stages=2,routed_input=True,output_dtype='bfloat16'),x=encoded,activation_scales=scales,w=graph.buffers['down'],scales=graph.buffers['down_scales'],**shared,out=graph.buffers['out'])
 return lower(graph,policy={'moe':dict(packed=True,packed_minimum_rows=1)})[0]

if args.compile_only:
 for rows in (2048,8192):
  for index,candidate in enumerate(candidates):
   produce(compose(rows,candidate),root/f'r{rows}-candidate{index}',search=False,target='sm_89')
   print('COMPILED',rows,index,candidate,flush=True)
 raise SystemExit(0)

metadata=dict(runtime='akas',purpose='expert-chain-benchmark',provider='cuda',
 model='LiquidAI/LFM2.5-24B-A2B-FP8',model_revision='f54bf39547fd5b7955abb9fe8d1b151e31fbbddb',
 qualification_scope='Original expert banks; complete packing, paired projection, optional quantizer and down projection; recorded errors versus block-result scaling; selected FP16 operands have independent operation qualification; not model throughput')
rec=Recorder(main/'benchmarks',metadata,source_root=work,kind='benchmark',change='Measure staged FP16 scaling with contiguous expert input tiles',dependency_sources={'tensor':main.parent/'tensor'})
rec.artifact(Path(__file__),'benchmark.py');print('STARTED',rec.id,flush=True);results=[]
try:
 cp=LFM2MoE.checkpoint_type(main/'build/checkpoints/lfm2.5-24b-a2b-fp8',tokenizer=False)
 names=['blk.2.ffn_'+part+'_exps.weight' for part in ('gate','up','down')];groups=names+[cp.scales[name] for name in names]
 wanted=tuple(name for name,layout in cp.layouts.items() if layout[0] in groups)
 with tensor.Device() as device,ExitStack() as lifetime:
  from ctypes import c_int,byref
  limit=c_int();device.driver.call('cuDeviceGetAttribute',byref(limit),97,device.device);shared_limit=limit.value
  bank=upload_shards(device,cp.tensors,wanted,cp.layouts)
  for value in bank.values():lifetime.callback(value.release)
  prepare,conditioning=lifetime.enter_context(cuda_working_set(device,256<<20,out=root/'working-set',target='sm_89'))
  for rows,case_index in ((2048,0),(8192,1)):
   profile=json.loads((main/f'build/lfm-serving/extended-serving-profile/profile.case-{case_index}.json').read_text())
   counts=np.array(profile['observations'][0]['outputs']['counts']['values'],dtype='int32')
   routes=np.full((64,rows),-1,'int32');filled=np.zeros(rows,'int32')
   for expert in np.argsort(-counts,kind='stable'):
    for offset,row in enumerate(sorted(np.argsort(filled,kind='stable')[:counts[expert]])):
     routes[expert,offset]=row*4+filled[row];filled[row]+=1
   assert (filled==4).all()
   tables=tile_tables(counts,rows+64,6);rng=np.random.default_rng(8017+rows)
   host=dict(x=encode_storage(rng.normal(0,96,(rows,2048)).astype('float32')),activation_scales=rng.uniform(.001,.01,(rows,16)).astype('float32'),counts=counts,routes=routes,tile_experts=tables[0],tile_offsets=tables[1])
   with ExitStack() as case:
    buffers={name:device.from_numpy(array) for name,array in host.items()};buffers['out']=device.zeros((rows,4,2048),'bfloat16')
    for value in buffers.values():case.callback(value.release)
    for key,name in zip(('w','w2','down'),names):buffers[key]=bank[name];buffers[key+'_scales']=bank[cp.scales[name]]
    reference=None
    for index,candidate in enumerate(candidates):
     destination=root/f'r{rows}-candidate{index}';produce(compose(rows,candidate),destination,search=False,target='sm_89')
     shared=[]
     for artifact in destination.glob('*.tbin'):
      with zipfile.ZipFile(artifact) as bundle:shared.append(json.loads(bundle.read('manifest.json'))['launch']['shared_memory_bytes'])
     if max(shared)>shared_limit:
      results.append(dict(rows=rows,candidate=candidate,status='rejected',reason='shared_memory',shared_memory_bytes=max(shared)))
      print('SKIP_SHARED',rows,index,max(shared),flush=True)
      continue
     with Forward(destination,device,buffers,graphs=True) as forward:
      forward.run();actual=buffers['out'].to_numpy()
      if reference is None:reference=actual.copy()
      else:assert np.isfinite(actual).all()
      timing=measure(device,forward.run,warmup=2,samples=7,repeats=5,reset=prepare)
     row=dict(rows=rows,candidate=candidate,bitwise=bool(np.array_equal(actual,reference)),maximum_absolute_error=float(np.max(np.abs(actual-reference))),relative_rms_error=float(np.linalg.norm(actual-reference)/np.linalg.norm(reference)),timing=timing);results.append(row)
     path=root/f'r{rows}-candidate{index}-result.json';path.write_text(json.dumps(row,indent=2)+'\n');rec.artifact(path,path.name);rec.artifact(destination/'forward.json',f'r{rows}-candidate{index}.json')
     print('RESULT',rows,index,candidate,timing['median_seconds'],flush=True)
  path=root/'results.json';path.write_text(json.dumps(dict(results=results,cache_conditioning=conditioning),indent=2)+'\n');rec.artifact(path,'results.json')
  for path in root.rglob('*.tbin'):rec.artifact(path,'producer/'+str(path.relative_to(root)))
 rec.update(dict(metadata=metadata,results=[]),final=True,status='passed');print('RUN',rec.id,flush=True)
except BaseException as error:
 if rec.record['completed_at'] is None:rec.update(dict(metadata=metadata,results=[],error=repr(error)),final=True,status='failed')
 raise
