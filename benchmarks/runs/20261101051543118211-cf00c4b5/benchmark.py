"""Original-weight paired projection, complete activation/quantization sequence."""
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
root=work/'build/lfm-serving/real-paired-reductions-v2';root.mkdir(parents=True,exist_ok=True)
meta=dict(runtime='akas',purpose='expert-reduction-sequence-benchmark',provider='cuda',model='LiquidAI/LFM2.5-24B-A2B-FP8',model_revision='f54bf39547fd5b7955abb9fe8d1b151e31fbbddb',qualification_scope='Original checkpoint gate/up banks, independent equations, complete captured paired projection, activation, quantization sequence; not model throughput')
rec=Recorder(main/'benchmarks',meta,source_root=work,kind='benchmark',change='Measure shared paired reduction partitions under original FP8 gate/up weight data',dependency_sources={'tensor':main.parent/'tensor'})
rec.artifact(Path(__file__),'benchmark.py');print('STARTED',rec.id,flush=True)
results=[]
try:
 checkpoint=LFM2MoE.checkpoint_type(main/'build/checkpoints/lfm2.5-24b-a2b-fp8',tokenizer=False)
 names=['blk.2.ffn_'+part+'_exps.weight' for part in ('gate','up')]
 groups=names+[checkpoint.scales[name] for name in names]
 wanted=tuple(n for n,layout in checkpoint.layouts.items() if layout[0] in groups)
 with tensor.Device() as device,ExitStack() as lifetime:
  bank=upload_shards(device,checkpoint.tensors,wanted,checkpoint.layouts)
  for value in bank.values():lifetime.callback(value.release)
  prepare,conditioning=lifetime.enter_context(cuda_working_set(device,256<<20,out=root/'working-set',target='sm_89'))
  weights=[bank[name].to_numpy() for name in names];scales=[bank[checkpoint.scales[name]].to_numpy() for name in names]
  meta['weight_banks']=[dict(name=name,weight_bits_sha256=hashlib.sha256(w.data).hexdigest(),scale_bits_sha256=hashlib.sha256(encode_bfloat16(s).data).hexdigest()) for name,w,s in zip(names,weights,scales)]
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
   x=encode_storage(rng.normal(0,96,(rows,2048)).astype('float32'))
   activation_scales=rng.uniform(.001,.01,(rows,16)).astype('float32')
   decoded=decode_storage(x).astype('float64')*np.repeat(activation_scales.astype('float64'),128,axis=-1)
   products=[]
   for w,s in zip(weights,scales):
    result=np.zeros((rows,4,1536),'float32')
    for expert,count in enumerate(counts):
     if not count:continue
     owners,ranks=np.divmod(routes[expert,:count],4)
     matrix=decode_storage(w[expert]).astype('float64')*np.repeat(np.repeat(s[expert].astype('float64'),128,axis=0),128,axis=1)
     result[owners,ranks]=(decoded[owners]@matrix.T).astype('float32')
    products.append(decode_bfloat16(encode_bfloat16(result)))
   g,u=products;expected=decode_bfloat16(encode_bfloat16(g.astype('float64')/(1+np.exp(-g.astype('float64')))*u))
   host=dict(x=x,activation_scales=activation_scales,counts=counts,routes=routes,tile_experts=tile_experts,tile_offsets=tile_offsets)
   with ExitStack() as case:
    buffers={n:device.from_numpy(v) for n,v in host.items()}
    buffers.update(w=bank[names[0]],scales=bank[groups[2]],w2=bank[names[1]],scales2=bank[groups[3]])
    for n,shape,dtype in [('out',(rows,4,1536),'bfloat16'),('encoded',(rows,4,1536),'uint8'),('output_scales',(rows,4,12),'float32')]:
     buffers[n]=device.zeros(shape,dtype);case.callback(buffers[n].release)
    for n in host:case.callback(buffers[n].release)
    candidates=[(1,32,128,1,'wide'),(1,64,128,1,'wide'),(1,128,128,1,'wide'),(1,128,128,1,'separate'),(2,64,128,1,'wide'),(4,64,128,1,'wide'),(4,128,128,1,'wide'),(4,128,128,1,'separate'),(8,64,128,1,'wide'),(8,128,128,1,'wide'),(16,128,128,1,'wide'),(4,128,256,1,'wide'),(2,64,128,2,'wide'),(1,32,128,2,'wide'),(1,64,128,2,'wide')]
    candidates=[(16,*candidate) for candidate in candidates]+[(32,1,32,128,2,'wide'),(32,1,64,128,1,'wide')]
    for index,(block_m,parts,columns,threads,stages,layout) in enumerate(candidates):
     graph=Graph(contract='reference')
     for n,v in buffers.items():graph.buffer(n,v.shape,str(v.dtype),role='output' if n in ('out','encoded','output_scales') else 'weight' if n in ('w','scales','w2','scales2') else 'input')
     p=dict(r=rows,rows=rows,top=4,experts=64,k=2048,o=1536,routed_input=False,compact=True,tile_map=True,max_block_m=128,block_m=block_m,partitions=parts,columns=columns,threads=threads,stages=stages,activation_vector=16,mma_dtype='float16',pair_layout=layout,product_dtype='bfloat16',activation_dtype='bfloat16')
     inputs={n:graph.buffers[n] for n in ('x','activation_scales','w','scales','w2','scales2','counts','routes','tile_experts','tile_offsets')}
     if parts==1:
      graph('expert_swiglu',dict(p,input_dtype='uint8',weight_format='fp8',output_dtype='bfloat16'),**inputs,out=graph.buffers['out'])
     else:
      gate=graph.buffer('gate',(rows,4,parts,1536),'float32');up=graph.buffer('up',gate.shape,'float32')
      graph('expert_pair_partial',p,**inputs,gate=gate,up=up)
      graph('swiglu_experts',dict(r=rows,top=4,c=1536,partitions=parts,input_dtype='float32',output_dtype='bfloat16',round_inputs=True,round_dtype='bfloat16'),g=gate,u=up,out=graph.buffers['out'])
     graph('quantize',dict(r=rows,top=4,k=1536,routed_input=True,input_dtype='bfloat16'),x=graph.buffers['out'],out=graph.buffers['encoded'],scales=graph.buffers['output_scales'])
     destination=root/f'r{rows}-candidate{index}';produce(graph,destination,search=False,target='sm_89')
     with Forward(destination,device,buffers,graphs=True) as forward:
      forward.run();actual=buffers['out'].to_numpy();np.testing.assert_allclose(actual,expected,rtol=.02,atol=.002)
      scales_expected=np.maximum(np.abs(actual).reshape(rows,4,12,128).max(-1),np.float32(1e-12))/np.float32(448)
      bits_expected=encode_storage(actual/np.repeat(scales_expected,128,axis=-1))
      np.testing.assert_array_equal(buffers['encoded'].to_numpy(),bits_expected)
      np.testing.assert_array_equal(buffers['output_scales'].to_numpy(),scales_expected)
      timing=measure(device,forward.run,warmup=2,samples=7,repeats=5,reset=prepare)
      row=dict(rows=rows,block_m=block_m,parts=parts,columns=columns,threads=threads,stages=stages,pair_layout=layout,independent_reference=True,timing=timing)
      results.append(row)
      retained=root/f'r{rows}-candidate{index}-result.json';retained.write_text(json.dumps(row,indent=2)+'\n');rec.artifact(retained,retained.name)
      print('RESULT',rows,block_m,parts,columns,threads,stages,layout,timing['median_seconds'],flush=True)
     rec.artifact(destination/'forward.json',f'r{rows}-candidate{index}.json')
  path=root/'results.json';path.write_text(json.dumps(dict(results=results,cache_conditioning=conditioning),indent=2)+'\n');rec.artifact(path,'results.json')
  for path in root.rglob('*.tbin'):rec.artifact(path,'producer/'+str(path.relative_to(root)))
 rec.update(dict(metadata=meta,results=[]),final=True,status='passed');print('RUN',rec.id,flush=True)
except BaseException as error:
 if rec.record['completed_at'] is None:rec.update(dict(metadata=meta,results=[],error=repr(error)),final=True,status='failed')
 raise
