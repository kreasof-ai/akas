"""Original-weight complete expert chains, including route packing overhead."""
from pathlib import Path
from contextlib import ExitStack
import hashlib,json
import numpy as np,tensor
from akas.models.lfm2 import LFM2MoE
from akas.checkpoints.upload import upload_shards
from akas.quantization._shared.e4m3 import encode_storage
from akas.ops.moe.validation import tile_tables
from akas.quantization.weights.fp8.kernel import prepare_host
from akas.graph import Graph
from akas.compilation.policy import lower
from akas.compilation.compile import produce
from akas.execution.forward import Forward
from akas.benchmarking.history import Recorder
from tensor.compiler.tuning import cuda_working_set,measure
main=Path('/home/sagemaker-user/akas');work=Path.cwd();root=work/'build/lfm-serving/real-joint-scale-chain';root.mkdir(parents=True,exist_ok=True)
metadata=dict(runtime='akas',purpose='expert-chain-benchmark',provider='cuda',model='LiquidAI/LFM2.5-24B-A2B-FP8',model_revision='f54bf39547fd5b7955abb9fe8d1b151e31fbbddb',qualification_scope='Original gate/up/down banks; complete captured expert chain including packing; numerical comparison with sequential block scaling and independent operation qualifications; not model throughput')
rec=Recorder(main/'benchmarks',metadata,source_root=work,kind='benchmark',change='Measure baseline-compatible joint FP8 block scales in complete expert chains',dependency_sources={'tensor':main.parent/'tensor'})
rec.artifact(Path(__file__),'benchmark.py');print('STARTED',rec.id,flush=True)
results=[]
try:
 cp=LFM2MoE.checkpoint_type(main/'build/checkpoints/lfm2.5-24b-a2b-fp8',tokenizer=False)
 names=['blk.2.ffn_'+part+'_exps.weight' for part in ('gate','up','down')];groups=names+[cp.scales[name] for name in names]
 wanted=tuple(n for n,layout in cp.layouts.items() if layout[0] in groups)
 with tensor.Device() as device,ExitStack() as lifetime:
  bank=upload_shards(device,cp.tensors,wanted,cp.layouts)
  for value in bank.values():lifetime.callback(value.release)
  blocked={}
  for name in names:
   prepared=prepare_host(bank[name].to_numpy(),bank[cp.scales[name]].to_numpy(),provider='cuda',storage_layout='block128')
   blocked[name]=device.from_numpy(prepared['values']);lifetime.callback(blocked[name].release)
  del prepared
  prepare,conditioning=lifetime.enter_context(cuda_working_set(device,256<<20,out=root/'working-set',target='sm_89'))
  metadata['weight_banks']=names
  for rows,case_index in ((2048,0),(8192,1)):
   profile_path=main/f'build/lfm-serving/extended-serving-profile/profile.case-{case_index}.json';profile=json.loads(profile_path.read_text());rec.artifact(profile_path,f'r{rows}-routing-profile.json')
   counts=np.array(profile['observations'][0]['outputs']['counts']['values'],dtype='int32')
   routes=np.full((64,rows),-1,'int32');filled=np.zeros(rows,'int32')
   for expert in np.argsort(-counts,kind='stable'):
    for offset,row in enumerate(sorted(np.argsort(filled,kind='stable')[:counts[expert]])):
     routes[expert,offset]=row*4+filled[row];filled[row]+=1
   assert (filled==4).all()
   tile_experts,tile_offsets=tile_tables(counts,(rows*4+3)//4+64,6)
   rng=np.random.default_rng(8017+rows)
   host=dict(x=encode_storage(rng.normal(0,96,(rows,2048)).astype('float32')),activation_scales=rng.uniform(.001,.01,(rows,16)).astype('float32'),counts=counts,routes=routes,tile_experts=tile_experts,tile_offsets=tile_offsets)
   with ExitStack() as case:
    buffers={name:device.from_numpy(array) for name,array in host.items()}
    for value in buffers.values():case.callback(value.release)
    for key,name in zip(('w','w2','down'),names):buffers[key]=bank[name];buffers[key+'_scales']=bank[cp.scales[name]]
    buffers['out']=device.zeros((rows,4,2048),'bfloat16');case.callback(buffers['out'].release)
    candidates=[(storage,scale_product,m,threads,stages,layout) for storage in ('raw','block128') for scale_product in ('sequential','joint') for m,threads,stages,layout in ((64,256,2,'wide'),(128,512,2,'separate'))]
    reference=None
    for index,(storage,scale_product,bm,threads,stages,layout) in enumerate(candidates):
     packed=True;down_m=bm;down_threads=threads;down_stages=stages
     for key,name in zip(('w','w2','down'),names):buffers[key]=(blocked if storage=='block128' else bank)[name]
     graph=Graph(contract='reference')
     for name,value in buffers.items():graph.buffer(name,value.shape,str(value.dtype),role='output' if name=='out' else 'input')
     activated=graph.buffer('activated',(rows,4,1536),'uint8');activated_scales=graph.buffer('activated_scales',(rows,4,12),'float32')
     shared={name:graph.buffers[name] for name in ('counts','routes','tile_experts','tile_offsets')}
     common=dict(r=rows,rows=rows,top=4,experts=64,partitions=1,compact=True,tile_map=True,max_block_m=128,columns=128,activation_vector=16,mma_dtype='float8_e4m3fn',weight_layout=storage,scale_product=scale_product)
     graph('fp8_expert_swiglu',dict(common,k=2048,o=1536,block_m=bm,threads=threads,stages=stages,pair_layout=layout,product_dtype='bfloat16',activation_dtype='bfloat16'),x=graph.buffers['x'],activation_scales=graph.buffers['activation_scales'],w=graph.buffers['w'],scales=graph.buffers['w_scales'],w2=graph.buffers['w2'],scales2=graph.buffers['w2_scales'],**shared,out=activated,output_scales=activated_scales)
     graph('expert_partial',dict(common,k=1536,o=2048,block_m=down_m,threads=down_threads,stages=down_stages,routed_input=True,output_dtype='bfloat16'),x=activated,activation_scales=activated_scales,w=graph.buffers['down'],scales=graph.buffers['down_scales'],**shared,out=graph.buffers['out'])
     if packed:graph,_=lower(graph,policy={'moe':dict(packed=True,packed_minimum_rows=1)})
     destination=root/f'r{rows}-candidate{index}';produce(graph,destination,search=False,target='sm_89')
     with Forward(destination,device,buffers,graphs=True) as forward:
      forward.run();actual=buffers['out'].to_numpy()
      if reference is None:reference=actual.copy()
      else:np.testing.assert_allclose(actual,reference,rtol=.02,atol=.002)
      timing=measure(device,forward.run,warmup=2,samples=7,repeats=5,reset=prepare)
      row=dict(rows=rows,weight_layout=storage,packed=packed,block_m=bm,threads=threads,stages=stages,pair_layout=layout,down_block_m=down_m,down_threads=down_threads,down_stages=down_stages,scale_product=scale_product,maximum_absolute_error=float(np.max(np.abs(actual-reference))),relative_rms_error=float(np.linalg.norm(actual-reference)/np.linalg.norm(reference)),timing=timing)
      results.append(row);path=root/f'r{rows}-candidate{index}-result.json';path.write_text(json.dumps(row,indent=2)+'\n');rec.artifact(path,path.name)
      print('RESULT',rows,storage,scale_product,packed,bm,threads,stages,layout,down_m,down_threads,down_stages,timing['median_seconds'],flush=True)
     rec.artifact(destination/'forward.json',f'r{rows}-candidate{index}.json')
  path=root/'results.json';path.write_text(json.dumps(dict(results=results,cache_conditioning=conditioning),indent=2)+'\n');rec.artifact(path,'results.json')
  for path in root.rglob('*.tbin'):rec.artifact(path,'producer/'+str(path.relative_to(root)))
 rec.update(dict(metadata=metadata,results=[]),final=True,status='passed');print('RUN',rec.id,flush=True)
except BaseException as error:
 if rec.record['completed_at'] is None:rec.update(dict(metadata=metadata,results=[],error=repr(error)),final=True,status='failed')
 raise
