"""Compare a real-weight routed block on recorded GPU inputs to independent equations."""
from pathlib import Path
import argparse,json,time
import numpy as np
from akas.models.lfm2.weights import LFM2MoECheckpoint
from support.native_fp8_reference import NativeFP8Reference
from akas.benchmarking.history import Recorder,digest
from akas.quantization._shared.e4m3 import decode_storage
parser=argparse.ArgumentParser();parser.add_argument('--trace',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
metadata=dict(runtime='numpy',purpose='local-numerical-trace',provider='cpu',model='LiquidAI/LFM2.5-24B-A2B-FP8',trace_sha256=digest(args.trace),qualification_scope='Independent real-weight block equations on recorded GPU inputs; no model acceptance claim')
record=Recorder('benchmarks',metadata,kind='diagnostic',change='Separate per-operation error from propagation through routed layers',dependency_sources={'tensor':Path('../tensor')})
record.artifact(Path(__file__),'check-local-moe.py');record.artifact(args.trace,'input-trace.npz');print('STARTED',record.id,flush=True)
started=time.monotonic();arrays=np.load(args.trace)
cp=LFM2MoECheckpoint('build/checkpoints/lfm2.5-24b-a2b-fp8',tokenizer=False);oracle=NativeFP8Reference(cp,projection_half=True)
try:
 layer=2;prefix=f'blk.{layer}.';x=arrays[prefix+'ffn_normalized'].reshape(5,2048)
 router=oracle.project(prefix+'ffn_gate_inp.weight',x);actual=arrays[prefix+'router'].reshape(5,64)
 def difference(a,b):return dict(relative_rms=float(np.linalg.norm(a-b)/np.linalg.norm(b)),max_error=float(abs(a-b).max()))
 result={'router':difference(actual,router)}
 for source,weight,target in (
  ('normalized','shortconv.in_proj.weight','projection'),
  ('ffn_normalized','ffn_gate.weight','gate'),
  ('ffn_normalized','ffn_up.weight','up'),
  ('activated','ffn_down.weight','ffn_output')):
  values=arrays['blk.0.'+source].reshape(5,-1)
  expected=oracle.project('blk.0.'+weight,values)
  result['dense0.'+target]=difference(arrays['blk.0.'+target].reshape(expected.shape),expected)
 score=1/(1+np.exp(-router));ids=np.argsort(-(score+oracle.weights[prefix+'exp_probs_b.bias']),axis=-1,kind='stable')[:,:4]
 actual_ids=arrays[prefix+'moe.ids'].reshape(5,4);result['ids_equal']=np.array_equal(actual_ids,ids);result['actual_ids']=actual_ids.tolist();result['reference_ids']=ids.tolist()
 if result['ids_equal']:
  probability=np.take_along_axis(score,ids,axis=1);probability/=probability.sum(axis=1,keepdims=True)+1e-6
  result['probability']=difference(arrays[prefix+'moe.probability'].reshape(5,4),probability)
 expected=oracle.feed_forward(layer,x,oracle.project);actual=arrays[prefix+'moe.out'].reshape(5,2048);result['moe_output']=difference(actual,expected)
 print(result,'elapsed',time.monotonic()-started,flush=True)
 args.out.write_text(json.dumps(result,indent=2)+'\n');record.artifact(args.out,'comparison.json');record.update(dict(metadata=metadata,results=[]),final=True,status='passed');print('RUN',record.id,flush=True)
except BaseException as error:
 if record.record['completed_at'] is None:record.update(dict(metadata=metadata,results=[],error=repr(error)),final=True,status='failed')
 raise
