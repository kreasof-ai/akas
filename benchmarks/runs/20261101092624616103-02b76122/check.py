"""Qualify opt-in compiler arithmetic without weakening independent checks."""
from pathlib import Path
import tensor
from tensor.compiler.cuda.nvrtc import NvrtcCompiler
from akas.benchmarking.history import Recorder
from test_lfm_moe_native import test_native_operations_match_independent_references as operation
from test_lfm_moe_native import test_native_model_reordered_ragged_requests_reuse_common_executor as model_check
original_options=NvrtcCompiler.options
NvrtcCompiler.options=lambda self,target,includes:original_options(self,target,includes)+['--use_fast_math']
main=Path('/home/sagemaker-user/akas');root=main/'build/lfm-serving/fastmath-arithmetic-qualification';root.mkdir(exist_ok=True)
metadata=dict(runtime='akas',purpose='arithmetic-qualification',provider='cuda',compiler_experiment={'nvrtc_options':['--use_fast_math']},qualification_scope='Unchanged independent operation tolerances plus a two-layer model with ragged/reordered requests and state reuse; not production-model qualification')
rec=Recorder(main/'benchmarks',metadata,source_root=main,kind='qualification',change='Qualify fast compiler arithmetic through established independent numerical and request-state checks',dependency_sources={'tensor':main.parent/'tensor'})
rec.artifact(Path(__file__),'check.py');print('STARTED',rec.id,flush=True)
try:
 checks=[]
 for index,(kind,mma) in enumerate((('expert_partial','float8_e4m3fn'),('expert_partial','float16'),('fp8_expert_swiglu','float8_e4m3fn'),('fp8_expert_swiglu','float16'))):
  parameters=dict(r=33,rows=33,k=384,o=256,experts=7,top=3,partitions=1,block_m=32,max_block_m=128,compact=True,tile_map=True,columns=128,threads=256,stages=2,pair_layout='wide',activation_vector=16,packed_input=True,packed_output=kind=='fp8_expert_swiglu',routed_input=kind=='expert_partial',product_dtype='bfloat16',activation_dtype='bfloat16',mma_dtype=mma)
  destination=root/f'operation-{index}';destination.mkdir(exist_ok=True)
  operation(destination,kind,parameters);checks.append(dict(kind=kind,parameters=parameters,passed=True));print('OPERATION_PASSED',index,kind,mma,flush=True)
 destination=root/'model';destination.mkdir(exist_ok=True)
 model_check(destination,32,True,True,9,'bfloat16','bfloat16',attention_half=True,expert_mma_dtype='float16',attention_decode_half=True)
 rec.program(destination/'program')
 for path in root.rglob('*.tbin'):rec.artifact(path,'producer/'+str(path.relative_to(root)))
 for path in (destination/'weights').iterdir():
  if path.is_file():rec.artifact(path,'fixture/'+path.name)
 metadata['operation_checks']=checks;metadata['two_layer_model_passed']=True
 rec.update(dict(metadata=metadata,results=[]),final=True,status='passed');print('RUN',rec.id,flush=True)
except BaseException as error:
 if rec.record['completed_at'] is None:rec.update(dict(metadata=metadata,results=[],error=repr(error)),final=True,status='failed')
 raise
