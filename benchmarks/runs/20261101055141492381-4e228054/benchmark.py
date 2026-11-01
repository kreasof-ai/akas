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
root=work/'build/lfm-serving/native-reader-real-expert-prefill';root.mkdir(parents=True,exist_ok=True)
meta=dict(runtime='akas',purpose='expert-reduction-sequence-benchmark',provider='cuda',model='LiquidAI/LFM2.5-24B-A2B-FP8',model_revision='f54bf39547fd5b7955abb9fe8d1b151e31fbbddb',qualification_scope='Original checkpoint gate/up banks and independent rounded-operand equations, floating counterpart qualification plus timed fused serving epilogue; not model throughput')
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
  for rows,case_index in ((2048,0),(8192,1)):
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
    candidates=([(None,32,256,1,'wide')] if rows==2048 else [(None,128,512,2,'separate')])+[('float16',32,256,1,'separate'),('float16',128,512,1,'separate')]
    for index,(dequantize_dtype,block_m,threads,stages,layout) in enumerate(candidates):
     parts,columns=1,128
     graph=Graph(contract='reference')
     for n,v in buffers.items():graph.buffer(n,v.shape,str(v.dtype),role='output' if n in ('out','encoded','output_scales') else 'weight' if n in ('w','scales','w2','scales2') else 'input')
     p=dict(r=rows,rows=rows,top=4,experts=64,k=2048,o=1536,routed_input=False,compact=True,tile_map=True,max_block_m=128,block_m=block_m,partitions=parts,columns=columns,threads=threads,stages=stages,activation_vector=16,mma_dtype='float8_e4m3fn',dequantize_dtype=dequantize_dtype,pair_layout=layout,product_dtype='bfloat16',activation_dtype='bfloat16')
     inputs={n:graph.buffers[n] for n in ('x','activation_scales','w','scales','w2','scales2','counts','routes','tile_experts','tile_offsets')}
     graph('expert_swiglu',dict(p,input_dtype='uint8',weight_format='fp8',output_dtype='bfloat16'),**inputs,out=graph.buffers['out'])
     destination=root/f'r{rows}-candidate{index}-floating';produce(graph,destination,search=False,target='sm_89')
     from tensor.artifacts.format import read_artifact
     from ctypes import c_int,byref
     limit=c_int();device.driver.call('cuDeviceGetAttribute',byref(limit),97,device.device)
     manifests=[read_artifact(path)[0] for path in destination.glob('*.tbin')]
     required=max(m['launch']['shared_memory_bytes'] for m in manifests)
     if required>limit.value:
      row=dict(rows=rows,dequantize_dtype=dequantize_dtype,block_m=block_m,threads=threads,stages=stages,pair_layout=layout,status='resource-rejected',shared_memory_bytes=required,device_limit_bytes=limit.value)
      results.append(row);print('REJECTED',row,flush=True);continue
     with Forward(destination,device,buffers,graphs=False) as forward:
      forward.run();actual=buffers['out'].to_numpy()
      # Independent original equations are sufficient for the unchanged path;
      # rounded operands have their own reference below.
      expected_operands=expected
      if dequantize_dtype is not None:
       def rounding(a):
        return (decode_bfloat16(encode_bfloat16(a.astype('float32'))).astype('float64') if dequantize_dtype=='bfloat16' else a.astype('float32').astype('float16').astype('float64'))
       dx=rounding(decoded);projections=[]
       for w,s in zip(weights,scales):
        projected=np.zeros((rows,4,1536),'float32')
        for expert,count in enumerate(counts):
         if not count:continue
         owners,ranks=np.divmod(routes[expert,:count],4)
         matrix=rounding(decode_storage(w[expert]).astype('float64')*np.repeat(np.repeat(s[expert].astype('float64'),128,axis=0),128,axis=1))
         projected[owners,ranks]=(dx[owners]@matrix.T).astype('float32')
        projections.append(decode_bfloat16(encode_bfloat16(projected)))
       g,u=projections;expected_operands=decode_bfloat16(encode_bfloat16(g.astype('float64')/(1+np.exp(-g.astype('float64')))*u))
      np.testing.assert_allclose(actual,expected_operands,rtol=.02,atol=.002)
     scales_expected=np.maximum(np.abs(actual).reshape(rows,4,12,128).max(-1),np.float32(1e-12))/np.float32(448)
     bits_expected=encode_storage(actual/np.repeat(scales_expected,128,axis=-1))
     fused=Graph(contract='reference');fused_buffers={**buffers,'out':buffers['encoded']}
     for name,value in fused_buffers.items():fused.buffer(name,value.shape,str(value.dtype),role='input' if name not in ('out','output_scales') else 'output')
     fused('fp8_expert_swiglu',p,**{name:fused.buffers[name] for name in inputs},out=fused.buffers['out'],output_scales=fused.buffers['output_scales'])
     destination=root/f'r{rows}-candidate{index}-fused';produce(fused,destination,search=False,target='sm_89')
     manifests=[read_artifact(path)[0] for path in destination.glob('*.tbin')]
     required=max(m['launch']['shared_memory_bytes'] for m in manifests)
     if required>limit.value:
      row=dict(rows=rows,dequantize_dtype=dequantize_dtype,block_m=block_m,threads=threads,stages=stages,pair_layout=layout,status='resource-rejected',shared_memory_bytes=required,device_limit_bytes=limit.value)
      results.append(row);print('REJECTED',row,flush=True);continue
     with Forward(destination,device,fused_buffers,graphs=True) as forward:
      forward.run();np.testing.assert_array_equal(fused_buffers['out'].to_numpy(),bits_expected)
      np.testing.assert_array_equal(fused_buffers['output_scales'].to_numpy(),scales_expected)
      timing=measure(device,forward.run,warmup=2,samples=7,repeats=5,reset=prepare)
      row=dict(rows=rows,dequantize_dtype=dequantize_dtype,block_m=block_m,threads=threads,stages=stages,pair_layout=layout,independent_reference=True,status='passed',timing=timing)
      results.append(row);retained=root/f'r{rows}-candidate{index}-result.json';retained.write_text(json.dumps(row,indent=2)+'\n');rec.artifact(retained,retained.name)
      print('RESULT',rows,dequantize_dtype,block_m,threads,stages,layout,timing['median_seconds'],flush=True)
     rec.artifact(destination/'forward.json',f'r{rows}-candidate{index}.json')
  path=root/'results.json';path.write_text(json.dumps(dict(results=results,cache_conditioning=conditioning),indent=2)+'\n');rec.artifact(path,'results.json')
  for path in root.rglob('*.tbin'):rec.artifact(path,'producer/'+str(path.relative_to(root)))
 rec.update(dict(metadata=meta,results=[]),final=True,status='passed');print('RUN',rec.id,flush=True)
except BaseException as error:
 if rec.record['completed_at'] is None:rec.update(dict(metadata=meta,results=[],error=repr(error)),final=True,status='failed')
 raise
