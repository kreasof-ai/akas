"""Measure complete down-projection reductions with one original expert bank."""
from pathlib import Path
from contextlib import ExitStack
import hashlib,json
import numpy as np,tensor
from akas.models.lfm2 import LFM2MoE
from akas.checkpoints.upload import upload_shards
from akas.quantization._shared.e4m3 import encode_storage,decode_storage
from akas.ops.moe.validation import tile_tables
from akas.graph import Graph
from akas.compilation.compile import produce
from akas.execution.forward import Forward
from akas.benchmarking.history import Recorder
from tensor.compiler.tuning import cuda_working_set,measure
from tensor.runtime.dtypes import encode_bfloat16,decode_bfloat16

main=Path('/home/sagemaker-user/akas');work=Path.cwd()
root=work/'build/lfm-serving/real-expert-reductions';root.mkdir(parents=True,exist_ok=True)
meta=dict(runtime='akas',purpose='expert-reduction-sequence-benchmark',provider='cuda',model='LiquidAI/LFM2.5-24B-A2B-FP8',model_revision='f54bf39547fd5b7955abb9fe8d1b151e31fbbddb',qualification_scope='Original checkpoint down-projection bank, independent equations, complete captured projection/merge sequence; not model throughput')
rec=Recorder(main/'benchmarks',meta,source_root=work,kind='benchmark',change='Measure compact reduction partitions under original FP8 weight data',dependency_sources={'tensor':main.parent/'tensor'})
rec.artifact(Path(__file__),'benchmark.py');print('STARTED',rec.id,flush=True)
results=[]
try:
 checkpoint=LFM2MoE.checkpoint_type(main/'build/checkpoints/lfm2.5-24b-a2b-fp8',tokenizer=False)
 name='blk.2.ffn_down_exps.weight';scale_name=checkpoint.scales[name]
 wanted=tuple(n for n,layout in checkpoint.layouts.items() if layout[0] in (name,scale_name))
 with tensor.Device() as device,ExitStack() as lifetime:
  bank=upload_shards(device,checkpoint.tensors,wanted,checkpoint.layouts)
  for value in bank.values():lifetime.callback(value.release)
  prepare,conditioning=lifetime.enter_context(cuda_working_set(device,256<<20,out=root/'working-set',target='sm_89'))
  w,scales=bank[name].to_numpy(),bank[scale_name].to_numpy()
  meta.update(weight_bank=name,weight_bits_sha256=hashlib.sha256(w.data).hexdigest(),scale_bits_sha256=hashlib.sha256(encode_bfloat16(scales).data).hexdigest())
  for rows,case_index in ((1,2),(4,3)):
   profile_path=main/f'build/lfm-serving/extended-serving-profile/profile.case-{case_index}.json'
   profile=json.loads(profile_path.read_text());rec.artifact(profile_path,f'r{rows}-routing-profile.json')
   counts=np.array(profile['observations'][0]['outputs']['counts']['values'],dtype='int32')
   routes=np.full((64,rows),-1,'int32');filled=np.zeros(rows,'int32')
   for expert in np.argsort(-counts,kind='stable'):
    selected=sorted(np.argsort(filled,kind='stable')[:counts[expert]])
    for offset,row in enumerate(selected):routes[expert,offset]=row*4+filled[row];filled[row]+=1
   assert (filled==4).all() and (counts<=rows).all()
   capacity=(rows*4+3)//4+64;tile_experts,tile_offsets=tile_tables(counts,capacity,6)
   rng=np.random.default_rng(1817+rows)
   x=encode_storage(rng.normal(0,96,(rows,4,1536)).astype('float32'))
   activation_scales=rng.uniform(.001,.01,(rows,4,12)).astype('float32')
   decoded=decode_storage(x).astype('float64')*np.repeat(activation_scales.astype('float64'),128,axis=-1)
   expected=np.zeros((rows,4,2048),'float32')
   for expert,count in enumerate(counts):
    if not count:continue
    owners,ranks=np.divmod(routes[expert,:count],4)
    weights=decode_storage(w[expert]).astype('float64')*np.repeat(np.repeat(scales[expert].astype('float64'),128,axis=0),128,axis=1)
    expected[owners,ranks]=(decoded[owners,ranks]@weights.T).astype('float32')
   expected=decode_bfloat16(encode_bfloat16(expected))
   host=dict(x=x,activation_scales=activation_scales,counts=counts,routes=routes,tile_experts=tile_experts,tile_offsets=tile_offsets)
   with ExitStack() as case:
    buffers={n:device.from_numpy(v) for n,v in host.items()}
    buffers.update(w=bank[name],scales=bank[scale_name])
    out=device.zeros((rows,4,2048),'bfloat16');buffers['out']=out;case.callback(out.release)
    for n in host:case.callback(buffers[n].release)
    for index,(parts,columns,threads,stages) in enumerate([(1,64,128,1),(2,64,128,1),(3,64,128,1),(4,64,128,1),(2,128,128,1),(3,128,128,1),(4,128,128,1),(3,128,256,1),(4,64,128,2)]):
     graph=Graph(contract='reference')
     for n,v in buffers.items():graph.buffer(n,v.shape,str(v.dtype),role='output' if n=='out' else 'weight' if n in ('w','scales') else 'input')
     p=dict(r=rows,rows=rows,top=4,experts=64,k=1536,o=2048,routed_input=True,compact=True,tile_map=True,max_block_m=128,block_m=16,partitions=parts,columns=columns,threads=threads,stages=stages,activation_vector=16,mma_dtype='float16',output_dtype='float32' if parts>1 else 'bfloat16')
     partial=graph.buffer('partial',(rows,4,parts,2048),'float32') if parts>1 else graph.buffers['out']
     graph('expert_partial',p,**{n:graph.buffers[n] for n in ('x','activation_scales','w','scales','counts','routes','tile_experts','tile_offsets')},out=partial)
     if parts>1:graph('expert_merge',dict(r=rows,top=4,o=2048,partitions=parts,output_dtype='bfloat16'),partial=partial,out=graph.buffers['out'])
     destination=root/f'r{rows}-candidate{index}';produce(graph,destination,search=False,target='sm_89')
     with Forward(destination,device,buffers,graphs=True) as forward:
      forward.run();actual=out.to_numpy();np.testing.assert_allclose(actual,expected,rtol=.02,atol=.002)
      timing=measure(device,forward.run,warmup=2,samples=7,repeats=5,reset=prepare)
      row=dict(rows=rows,parts=parts,columns=columns,threads=threads,stages=stages,independent_reference=True,timing=timing)
      results.append(row);print('RESULT',rows,parts,columns,threads,stages,timing['median_seconds'],flush=True)
     rec.artifact(destination/'forward.json',f'r{rows}-candidate{index}.json')
  path=root/'results.json';path.write_text(json.dumps(dict(results=results,cache_conditioning=conditioning),indent=2)+'\n');rec.artifact(path,'results.json')
  for path in root.rglob('*.tbin'):rec.artifact(path,'producer/'+str(path.relative_to(root)))
 rec.update(dict(metadata=meta,results=[]),final=True,status='passed');print('RUN',rec.id,flush=True)
except BaseException as error:
 if rec.record['completed_at'] is None:rec.update(dict(metadata=meta,results=[],error=repr(error)),final=True,status='failed')
 raise
