"""Natural-prompt quality probes through the measured SGLang configuration."""
from pathlib import Path
import argparse,json
import sglang
root=Path('/home/sagemaker-user/akas/build/lfm-serving')
parser=argparse.ArgumentParser()
parser.add_argument('--suite',type=Path,default=root/'reference-text-suite.json')
parser.add_argument('--out',type=Path,default=root/'sglang-text-probes.json')
args=parser.parse_args()
reference=json.loads(args.suite.read_text())
settings=dict(model_path='/home/sagemaker-user/akas/build/checkpoints/lfm2.5-24b-a2b-fp8',dtype='bfloat16',context_length=2184,max_running_requests=256,chunked_prefill_size=2048,mem_fraction_static=.85,disable_cuda_graph=False,enable_metrics=True,disable_radix_cache=True,random_seed=17,log_level='info',kv_cache_dtype='auto',cuda_graph_bs_decode=[1,4,16,64,256],skip_tokenizer_init=True,stream_interval=1,incremental_streaming_output=False)
sampling=dict(temperature=0,max_new_tokens=reference.get('generated_tokens',32),ignore_eos=True)
if __name__=='__main__':
 engine=sglang.Engine(**settings)
 rows=[]
 try:
  for row in reference['rows']:
   actual=engine.generate(input_ids=row['tokens'],sampling_params=sampling,return_logprob=True,top_logprobs_num=20)
   result=dict(index=row['index'],response=actual)
   if 'prediction' in row:result.update(reference_prediction=row['prediction'],greedy_equal=actual['output_ids'][0]==row['prediction'])
   rows.append(result);print('PROBE',row['index'],actual['output_ids'][0],flush=True)
   args.out.write_text(json.dumps(dict(version=sglang.__version__,settings=settings,sampling=sampling,rows=rows),indent=2)+'\n')
 finally:engine.shutdown()
