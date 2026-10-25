"""Retain cold-cache counters separately from native/HTTP throughput results."""
import csv,io,json,subprocess,traceback
from pathlib import Path
from akas.benchmarking.history import Recorder
root=Path('build/lfm-moe/expert-counters');root.mkdir(exist_ok=True);toolkit=Path('build/tools/nsight');ncu=(toolkit/'extracted/opt/nvidia/nsight-compute/2026.3.1/ncu').resolve();python=Path('.venv/bin/python').absolute();target=Path('build/lfm-moe/profile-expert-counters-target.py').resolve()
meta=dict(runtime='akas',purpose='kernel',protocol='cuda-counter-profile-v1',provider='cuda',target='sm_89',device=dict(name='NVIDIA L4',provider='cuda'),weight_format='q4_0',operand_dtype='float16',accumulator_dtype='float32',profiler_package=json.loads((toolkit/'package.json').read_text()),qualification='Diagnostic hardware counters only. Qualified AOT projections with deterministic random, unique top-4 routes; final row inactive. L2 flushed by profiler, clock control disabled; kernel replay. Packed candidate excludes preparation, already measured including packing separately. No serving-speed claim and no measured real model routes.')
rec=Recorder('benchmarks',meta,kind='diagnostic',change='Distinguish short and large-chunk expert prefill bottlenecks with hardware counters');report=dict(schema='akas.benchmark.v1',metadata=meta,results=[]);passed=False
try:
 checks=[]
 for rows,candidate in ((32,'baseline'),(4096,'baseline'),(4096,'packed')):
  label=f'{rows}-{candidate}';path=(root/f'{label}.csv').resolve();export=(root/label).resolve();argv=['sudo','-n','env','PYTHONDONTWRITEBYTECODE=1',str(ncu),'--clock-control','none','--cache-control','all','--replay-mode','kernel','--launch-count','1','--csv','--page','raw','--log-file',str(path),'-o',str(export)]
  for section in ('SpeedOfLight','MemoryWorkloadAnalysis','Occupancy','SchedulerStats','WarpStateStats'):argv+=['--section',section]
  argv+=[str(python),str(target),'--rows',str(rows),'--candidate',candidate];print('PROFILE',label,flush=True);result=subprocess.run(argv,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=120);(root/f'{label}.log').write_text(result.stdout);print(result.stdout[-1500:],flush=True)
  if result.returncode:raise RuntimeError(f'{label} profiler exited {result.returncode}: {result.stdout[-1500:]}')
  text=path.read_text();lines=text.splitlines();header=next(i for i,line in enumerate(lines) if '"Metric Name"' in line and '"Metric Value"' in line);metrics=[]
  for row in csv.DictReader(io.StringIO('\n'.join(lines[header:]))):
   if row.get('Metric Name'):metrics.append({k:row.get(k) for k in ('Section Name','Metric Name','Metric Unit','Metric Value')})
  checks.append(dict(rows=rows,candidate=candidate,metrics=metrics));meta['checks']=checks;print('COLLECTED',label,len(metrics),flush=True)
 meta['checks']=checks;passed=True
except BaseException:
 meta['traceback']=traceback.format_exc();print(meta['traceback'],flush=True)
finally:
 rec.artifact(__file__,'profiling/runner.py');rec.artifact(target,'profiling/target.py')
 for p in root.glob('*'):
  if p.is_file():rec.artifact(p,'profiling/'+p.name)
 rec.update(report,final=True,status='passed' if passed else 'failed');print('RETAINED',rec.id,passed,flush=True)
if not passed:raise SystemExit(1)
