"""Vendor GEMM and streaming-copy calibration, separate from model throughput."""
from pathlib import Path
import argparse,ctypes,json,statistics,subprocess
import numpy as np
import torch
from akas.benchmarking.history import Recorder

parser=argparse.ArgumentParser()
parser.add_argument('--counter-probe',action='store_true')
parser.add_argument('--extended',action='store_true')
args=parser.parse_args()
torch.cuda.init()
torch.backends.cuda.matmul.allow_tf32=False
torch.set_float32_matmul_precision('highest')
torch.manual_seed(91)
root=Path('build/lfm-serving/device-ceilings')
root.mkdir(parents=True,exist_ok=True)
nvml=ctypes.CDLL('libnvidia-ml.so.1')
assert nvml.nvmlInit_v2()==0
handle=ctypes.c_void_p()
assert nvml.nvmlDeviceGetHandleByIndex_v2(0,ctypes.byref(handle))==0

def clock_mhz():
 value=ctypes.c_uint()
 assert nvml.nvmlDeviceGetClockInfo(handle,1,ctypes.byref(value))==0
 return value.value

def measure(callback, *, warmup=10, samples=9, repeats=5):
 for _ in range(warmup):callback()
 torch.cuda.synchronize()
 start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
 values=[];clocks=[]
 for _ in range(samples):
  start.record()
  for _ in range(repeats):callback()
  end.record();end.synchronize()
  values.append(start.elapsed_time(end)/1000/repeats)
  clocks.append(clock_mhz())
 return dict(samples_seconds=values,median_seconds=statistics.median(values),sm_clock_mhz=clocks)

def gemm(dtype, size=8192, fast_accum=False):
 def operand():
  value=torch.randn((size,size),device='cuda',dtype=torch.float16)*.125
  # Exact binary products isolate the fast accumulator's compute throughput.
  # Its rejected normal-input numerical probe remains in the previous record.
  return (value.sign()*.125 if fast_accum else value).to(dtype)
 x,w=operand(),operand()
 output_dtype=torch.bfloat16 if dtype==torch.float8_e4m3fn else dtype
 out=torch.empty((size,size),device='cuda',dtype=output_dtype)
 scale=torch.ones((),device='cuda',dtype=torch.float32)
 if dtype==torch.float8_e4m3fn:
  callback=lambda:torch.ops.aten._scaled_mm.out(x,w.T,scale,scale,
            out_dtype=output_dtype,use_fast_accum=fast_accum,out=out)
 else:callback=lambda:torch.mm(x,w.T,out=out)
 if args.counter_probe:
  callback();torch.cuda.synchronize()
  torch.cuda.profiler.start()
  try:callback();torch.cuda.synchronize()
  finally:torch.cuda.profiler.stop()
  return None
 callback();torch.cuda.synchronize()
 expected=x[:8].float().cpu().numpy().astype('float64')@w[:8].float().cpu().numpy().astype('float64').T
 actual=out[:8,:8].float().cpu().numpy()
 tolerance=.008 if output_dtype==torch.bfloat16 else .002 if dtype==torch.float16 else 2e-5
 np.testing.assert_allclose(actual,expected,rtol=tolerance,atol=.002 if output_dtype==torch.bfloat16 else .0005)
 result=measure(callback)
 return dict(kind='dense_gemm',dtype=str(dtype),output_dtype=str(output_dtype),m=size,n=size,k=size,
    input_distribution='random signs times 0.125' if fast_accum else 'normal times 0.125',
    fma_counted_as_two_flops=True,use_fast_accum=fast_accum,flops=2*size**3,**result,
    teraflops_per_second=2*size**3/result['median_seconds']/1e12)

if args.counter_probe:
 gemm(torch.float8_e4m3fn)
 raise SystemExit(0)

metadata=dict(runtime='device-calibration',purpose='hardware-ceiling',provider='cuda',
 device=torch.cuda.get_device_name(),torch_version=torch.__version__,
 theoretical_dense_peaks=dict(fp8_tflops=733.,fp16_tflops=366.5,fp32_tflops=91.6,dram_gb_per_second=864.),
 specification_source='https://www.nvidia.com/en-sg/data-center/l40s/',
 qualification_scope='Vendor dense GEMM and streaming copy calibration; not model throughput')
record=Recorder('benchmarks',metadata,kind='benchmark',change='Measure compute and DRAM ceilings for inference optimization')
record.artifact(Path(__file__),'device-ceilings.py')
try:
 results=([gemm(torch.float8_e4m3fn,size=size,fast_accum=True) for size in (8192,16384)]
          if args.extended else [gemm(dtype) for dtype in (torch.float8_e4m3fn,torch.float16,torch.float32)])
 source=torch.randn((1<<30,),device='cuda',dtype=torch.float32)
 destination=torch.empty_like(source)
 callback=lambda:destination.copy_(source)
 callback();torch.cuda.synchronize()
 assert torch.equal(source[:1024],destination[:1024])
 if args.extended:
  scalar=torch.empty((),device='cuda',dtype=torch.float32)
  expected=source.double().sum().item()
  callback=lambda:torch.sum(source,dim=(0,),out=scalar)
  callback();torch.cuda.synchronize()
  np.testing.assert_allclose(scalar.item(),expected,rtol=2e-5,atol=.05)
  timing=measure(callback,warmup=3,repeats=3)
  traffic=source.numel()*source.element_size()
  results.append(dict(kind='streaming_read_reduction',read_bytes=traffic,
      working_set_exceeds_l2=True,**timing,gigabytes_per_second=traffic/timing['median_seconds']/1e9))
 else:
  timing=measure(callback,warmup=3,repeats=3)
  traffic=source.numel()*source.element_size()*2
  results.append(dict(kind='streaming_copy',source_bytes=source.numel()*source.element_size(),
      read_plus_write_bytes=traffic,working_set_exceeds_l2=True,**timing,
      gigabytes_per_second=traffic/timing['median_seconds']/1e9))
 report=dict(metadata=metadata,calibration=results,
   clocks=subprocess.run(['nvidia-smi','--query-gpu=clocks.current.sm,clocks.current.memory,power.draw,temperature.gpu',
     '--format=csv'],capture_output=True,text=True,check=True).stdout)
 path=root/('extended-results.json' if args.extended else 'results.json');path.write_text(json.dumps(report,indent=2)+'\n')
 record.artifact(Path(__file__),'device-ceilings.py')
 record.artifact(path,'device-ceilings.json')
 record.update(dict(metadata=metadata,results=[]),final=True,status='passed')
 print('RUN',record.id,flush=True)
 for result in results:print(result,flush=True)
except BaseException as error:
 if record.record['completed_at'] is None:
  record.update(dict(metadata=metadata,results=[],error=repr(error)),final=True,status='failed')
 raise
