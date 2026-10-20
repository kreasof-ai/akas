"""Diagnostic library bound; no library kernel is added to AKAS execution."""
import json,statistics,traceback
from pathlib import Path
import torch
from akas.benchmarking.history import Recorder
meta=dict(runtime='pytorch-cublas',purpose='kernel',protocol='kernel-latency-v1',provider='cuda',target='sm_89',device=dict(name='NVIDIA L4',provider='cuda'),operand_dtype='float16',accumulator_dtype='float32',qualification='Diagnostic regular batched dense GEMM upper bound only. Excludes decoding Q4, route gathering, irregular counts, and output scattering. F32 output and reduced-precision reduction disabled. Not a model or serving result.')
rec=Recorder('benchmarks',meta,kind='algorithm',change='Measure regular dense library bound before redesigning routed expert execution')
report=dict(schema='akas.benchmark.v1',metadata=meta,results=[]);passed=False
try:
 torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
 torch.manual_seed(17);checks=[]
 for rows in (256,1024):
  for depth,columns in ((2048,1536),(1536,2048)):
   x=torch.randn((64,rows,depth),device='cuda',dtype=torch.float16);w=torch.randn((64,columns,depth),device='cuda',dtype=torch.float16);out=torch.empty((64,rows,columns),device='cuda',dtype=torch.float32)
   def call():torch.bmm(x,w.transpose(1,2),out_dtype=torch.float32,out=out)
   for _ in range(20):call()
   torch.cuda.synchronize();samples=[]
   for _ in range(7):
    start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record()
    for _ in range(5):call()
    end.record();end.synchronize();samples.append(start.elapsed_time(end)/5/1000)
   median=statistics.median(samples);row=dict(experts=64,rows=rows,depth=depth,columns=columns,median_seconds=median,samples=samples,teraflops=2*64*rows*depth*columns/median/1e12);checks.append(row);print(json.dumps(row),flush=True)
   del x,w,out
 meta['checks']=checks;meta['torch_version']=torch.__version__;passed=True
except BaseException:
 meta['traceback']=traceback.format_exc();print(meta['traceback'],flush=True)
finally:
 rec.artifact(__file__,'kernels/torch-native-bound.py');rec.update(report,final=True,status='passed' if passed else 'failed');print('RETAINED',rec.id,passed,flush=True)
if not passed:raise SystemExit(1)
