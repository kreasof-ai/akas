"""Replay the independent model checks on an already-produced AOT fixture."""
from pathlib import Path
from unittest.mock import patch
import argparse,json
import akas.compilation
import test_lfm_moe_native as tests
from akas.benchmarking.history import Recorder

parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
parser.add_argument('--residual-dtype',choices=['float32','float16','bfloat16'],default='float32')
parser.add_argument('--half-decode',action='store_true');args=parser.parse_args()
main=Path('/home/sagemaker-user/akas');root=args.root.resolve();program=root/'program'
manifest=json.loads((program/'program.json').read_text());policy=manifest['policy']
assert policy.get('residual',{}).get('intermediate_dtype','float32')==args.residual_dtype
assert (policy.get('attention',{}).get('decode_precision')=='float16')==args.half_decode
for phase in manifest['phases']:
 data=json.loads((program/phase['path']).read_text())
 for kernel in data['kernels'].values():
  if kernel['kind'] in ('expert_partial','expert_swiglu','fp8_expert_swiglu'):
   assert kernel['schedule'].get('mma_dtype')=='float16'
metadata=dict(runtime='akas',purpose='model-numerical-qualification',provider='cuda',
 qualification_scope='Independent logits, reordered/ragged requests, retirement and state reuse on an existing AOT fixture; compilation is intentionally bypassed',
 residual_dtype=args.residual_dtype,attention_decode_half=args.half_decode,expert_mma_dtype='float16')
record=Recorder(main/'benchmarks',metadata,source_root=main,kind='qualification',
 change='Replay existing model precision and state checks on retained AOT fixtures',dependency_sources={'tensor':main.parent/'tensor'})
record.artifact(Path(__file__),'check.py');record.program(program)
for path in (root/'weights').iterdir():
 if path.is_file():record.artifact(path,'fixture/'+path.name)
print('STARTED',record.id,flush=True)
try:
 # The graph above is already compiled and archived. Reuse the established
 # numerical/state assertions without rebuilding it or regenerating weights.
 with patch.object(tests,'fixture',lambda path:path),patch.object(akas.compilation,'compile_checkpoint',lambda *a,**k:None):
  tests.test_native_model_reordered_ragged_requests_reuse_common_executor(root,32,True,True,9,
    'bfloat16','bfloat16',residual_dtype=args.residual_dtype,attention_half=True,
    expert_mma_dtype='float16',attention_decode_half=args.half_decode)
 record.update(dict(metadata=metadata,results=[]),final=True,status='passed');print('RUN',record.id,flush=True)
except BaseException as error:
 if record.record['completed_at'] is None:record.update(dict(metadata=metadata,results=[],error=repr(error)),final=True,status='failed')
 raise
