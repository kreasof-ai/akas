"""Diagnostic original-FP8 serving matrix through the common streamed protocol."""
from pathlib import Path
from dataclasses import asdict
import argparse,asyncio,hashlib,json,statistics,time
import numpy as np
import tensor
from akas import Engine,EngineConfig
from akas.models.lfm2 import LFM2MoE
from akas.benchmarking.cases import Case
from akas.benchmarking.history import Recorder,write
from akas.benchmarking.runner import summarize
from akas.benchmarking.session import Session
from akas.benchmarking.streaming import engine_burst
from akas.benchmarking.workload import synthetic_prompts,prompt_digest,prompt_seed
root=Path('build/lfm-serving')
parser=argparse.ArgumentParser()
parser.add_argument('--program',type=Path,default=root/'program-concurrent')
parser.add_argument('--profile',type=Path,default=root/'serving-profile.json')
parser.add_argument('--batches',type=int,nargs='+',default=[1,4,16,64,256])
parser.add_argument('--qualification',type=Path,action='append',default=[])
parser.add_argument('--diagnostic',type=Path,action='append',default=[])
parser.add_argument('--reference-run',default='20261031203613246653-4a984cbd')
parser.add_argument('--qualification-only',action='store_true')
parser.add_argument('--completion-probes',action='store_true')
parser.add_argument('--chat-probes',type=Path)
args=parser.parse_args()
program=args.program
reference_record=json.loads((Path('benchmarks/runs')/args.reference_run/'run.json').read_text())
reference_rows={row['case']['batch']:row for row in reference_record['results']}
config=EngineConfig(prefill_chunk=2048,prefill_requests=256,prefill_token_budget=256*2048,decode_prefill_token_budget=256)
metadata=dict(runtime='akas',purpose='concurrent-fp8-diagnostic',model='LiquidAI/LFM2.5-24B-A2B-FP8',model_revision='f54bf39547fd5b7955abb9fe8d1b151e31fbbddb',weight_format='fp8',provider='cuda',target='sm_89',mode='ar',kv_dtype='float16',kv_bits=16,prefix_cache=False,speculative_decoding=False,cpu_offload=False,early_eos=False,protocol='client-stream-burst-v1',full_model_numerical_qualification='open',model_throughput_qualified=False,engine_config=asdict(config),program_sha256=hashlib.sha256((program/'program.json').read_bytes()).hexdigest(),reference_run_id=args.reference_run,warmup=1,repeats=3)
precision=[]
for phase in json.loads((program/'program.json').read_text())['phases']:
 kernels=list(json.loads((program/phase['path']).read_text())['kernels'].values())
 quantized=[k for k in kernels if k['kind'] in ('projection_partial','expert_partial','expert_swiglu','fp8_expert_swiglu')]
 precision.append(dict(phase=phase['phase'],rows=phase['rows'],chunk=phase['chunk'],
  quantized_mma_dtypes=sorted({k['schedule'].get('mma_dtype',k['parameters'].get('mma_dtype','float8_e4m3fn')) for k in quantized}),
  expert_contribution_dtypes=sorted({k['parameters'].get('output_dtype','float32') for k in kernels if k['kind']=='expert_partial' and k['parameters'].get('routed_input')})))
 precision[-1]['expert_product_dtypes']=sorted({k['parameters'].get('product_dtype','float32') for k in kernels if k['kind'] in ('expert_swiglu','fp8_expert_swiglu')})
 precision[-1]['expert_activation_dtypes']=sorted({k['parameters'].get('activation_dtype','float32') for k in kernels if k['kind'] in ('expert_swiglu','fp8_expert_swiglu')})
metadata['execution_precision']=precision
rec=Recorder('/home/sagemaker-user/akas/benchmarks',metadata,kind='benchmark',change='Measure original FP8 multi-user serving with shared expert and projection policies; numerical acceptance remains open',dependency_sources={'tensor':Path('/home/sagemaker-user/tensor')})
rec.artifact(Path(__file__),'benchmark-concurrent.py')
print('STARTED',rec.id,flush=True)
results=[]
try:
 started=time.monotonic()
 with tensor.Device() as device:
  metadata['device']=device.info
  model=LFM2MoE('build/checkpoints/lfm2.5-24b-a2b-fp8',program,device,checkpoint_options={'tokenizer':False})
  print('LOADED',time.monotonic()-started,flush=True)
  # Compare the currently available independent natural-text probes outside timing.
  text_reference=root/'reference-text-suite.json'
  if text_reference.exists():
   rec.artifact(text_reference,'reference-text-suite.json')
   rec.artifact(root/'reference-text-suite.npz','reference-text-suite.npz')
   reference=json.loads(text_reference.read_text());arrays=np.load(root/'reference-text-suite.npz')
   checks=[]
   for row in reference['rows']:
    slot=model.admit()
    predicted=model.prefill([slot],[row['tokens']],chunk=2048)[0]
    actual=model.last_frame.feature('logits').to_numpy().reshape(-1)
    expected=arrays[f"logits{row['index']}"]
    check=dict(index=row['index'],prediction=predicted,reference_prediction=row['prediction'],greedy_equal=predicted==row['prediction'],correlation=float(np.corrcoef(actual,expected)[0,1]),rms_error=float(np.sqrt(np.mean((actual-expected)**2))))
    checks.append(check);print('TEXT',check,flush=True)
    model.release(slot)
   metadata['natural_text_probes']=checks
  with Engine(model,config=config) as engine:
   if args.chat_probes:
    rec.artifact(args.chat_probes,'chat-probes.json')
    chat=json.loads(args.chat_probes.read_text())
    checkpoint=LFM2MoE.checkpoint_type(root.parent/'checkpoints/lfm2.5-24b-a2b-fp8')
    rows=[]
    for probe in chat['rows']:
     ids,_=asyncio.run(engine_burst(engine,[probe['tokens']],chat['generated_tokens']))
     row=dict(index=probe['index'],output_ids=ids[0],text=checkpoint.tokenizer.decode(ids[0]))
     rows.append(row);print('CHAT',row,flush=True)
    metadata['chat_probes']=rows
   if args.completion_probes:
    checkpoint=LFM2MoE.checkpoint_type(root.parent/'checkpoints/lfm2.5-24b-a2b-fp8')
    completions=[]
    for probe in json.loads(text_reference.read_text())['rows']:
     ids,metrics=asyncio.run(engine_burst(engine,[probe['tokens']],32))
     row=dict(index=probe['index'],output_ids=ids[0],text=checkpoint.tokenizer.decode(ids[0]),
              reference_prediction=probe['prediction'],greedy_equal=ids[0][0]==probe['prediction'])
     completions.append(row);print('COMPLETION',row,flush=True)
    metadata['natural_text_completions']=completions
   for batch in (() if args.qualification_only else args.batches):
    prompts=synthetic_prompts(model.config.vocab,batch,2048)
    print('WARMUP',batch,flush=True)
    asyncio.run(engine_burst(engine,prompts,128))
    samples,outputs=[],[]
    for repeat in range(3):
     ids,metrics=asyncio.run(engine_burst(engine,prompts,128))
     metrics['decode_tokens_per_second_per_request']=metrics['decode_tokens_per_second']/batch
     metrics['inter_token_seconds']=metrics['tpot_seconds']
     metrics['wall_seconds']=max(stream['arrivals_seconds'][-1] for stream in metrics['request_streams'])
     samples.append(metrics);outputs.append(ids)
     print('SAMPLE',batch,repeat,{k:metrics[k] for k in ('ttft_seconds','prefill_tokens_per_second','decode_tokens_per_second')},flush=True)
    summary={key:summarize([sample[key] for sample in samples]) for key in ('ttft_seconds','prefill_tokens_per_second','decode_tokens_per_second','decode_tokens_per_second_per_request','inter_token_seconds','wall_seconds')}
    row=dict(case=Case(batch,2048,128).metadata(),status='ok',seed=prompt_seed(batch,2048),prompt_source='synthetic',prompt_sha256=prompt_digest(prompts),prompt_tokens=batch*2048,generated_tokens=batch*128,decode_tokens=batch*127,allocated_bytes=Session(model,batch=batch,chunk=2048).allocated_bytes,model_throughput_qualified=False,full_model_numerical_qualification='open',samples=samples,summary=summary,output_ids=outputs[0],deterministic=all(value==outputs[0] for value in outputs),scheduler=dict(engine.scheduler.stats))
    reference=reference_rows[batch]
    assert reference['prompt_sha256']==row['prompt_sha256']
    row['reference_output_equal']=outputs[0]==reference['output_ids']
    print('REFERENCE_OUTPUT_EQUAL',batch,row['reference_output_equal'],flush=True)
    results.append(row)
    rec.update(dict(metadata=metadata,results=results))
    write(root/'native-concurrent-diagnostic.json',dict(metadata=metadata,results=results,run_id=rec.id))
 rec.artifact(Path(__file__),'benchmark-concurrent.py')
 write(root/'native-concurrent-diagnostic.json',dict(metadata=metadata,results=results,run_id=rec.id))
 rec.artifact(program/'program.json','program.json')
 rec.program(program)
 rec.artifact(args.profile,'serving-profile.json')
 rec.artifact(root/'dense-projection-qualified/report.json','dense-projection-qualification.json')
 rec.artifact(root/'expert-vector-qualified/report.json','expert-projection-qualification.json')
 for index,path in enumerate(args.qualification):rec.artifact(path,f'additional-qualification-{index}.json')
 for index,path in enumerate(args.diagnostic):rec.artifact(path,f'diagnostic-{index}{path.suffix}')
 rec.artifact(root/'native-concurrent-diagnostic.json','raw.json')
 record=rec.update(dict(metadata=metadata,results=results),final=True,status='passed')
 print('RUN',rec.id,record['status'],flush=True)
except BaseException:
 rec.update(dict(metadata=metadata,results=results),final=True,status='failed')
 raise
