from pathlib import Path
import json

def main():
 import sglang
 rows=json.loads(Path('build/lfm-serving/natural-text-probes.json').read_text())
 engine=sglang.Engine(model_path='/home/sagemaker-user/akas/build/checkpoints/lfm2.5-24b-a2b-fp8',dtype='bfloat16',kv_cache_dtype='auto',context_length=2176,max_running_requests=16,chunked_prefill_size=2048,mem_fraction_static=.85,disable_cuda_graph=True,disable_radix_cache=True,skip_tokenizer_init=True,random_seed=17,log_level='warning')
 try:
  outputs=[]
  for row in rows:
   response=engine.generate(input_ids=row['tokens'],sampling_params=dict(temperature=0,max_new_tokens=1,ignore_eos=True),return_logprob=True,top_logprobs_num=10)
   result={**row,'response':response};outputs.append(result)
   print('PROBE',row['text'],row['bos'],response['output_ids'],flush=True)
   Path('build/lfm-serving/sglang-text-probes.json').write_text(json.dumps(dict(runtime='sglang',version=sglang.__version__,diagnostic=True,model_revision='f54bf39547fd5b7955abb9fe8d1b151e31fbbddb',kv_dtype='bfloat16',rows=outputs),indent=2)+'\n')
 finally:engine.shutdown()
if __name__=='__main__':main()
