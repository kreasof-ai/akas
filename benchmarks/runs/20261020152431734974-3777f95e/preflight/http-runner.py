"""Complete HTTP validation using a service constructed on its owner thread."""
import asyncio,copy,json,os,traceback
from contextlib import contextmanager
from pathlib import Path
import tensor
from akas.models.lfm2.model import LFM2MoE
from akas.engine import Engine
from akas.config import EngineConfig
from akas.serving.http import Service
from akas.optimization.evaluation import http_benchmark
from akas.benchmarking.history import Recorder,write,source_identity
parent_id='20261020145716182537-a52f9257'
parent=Path('benchmarks/runs')/parent_id
original=json.loads((parent/'run.json').read_text())
report=copy.deepcopy(json.loads((parent/'report.json').read_text()))
assert source_identity(Path.cwd())['tree_sha256']==original['source']['tree_sha256']
meta=report['metadata'];meta.pop('error',None);meta.pop('traceback',None)
meta.update(native_measurement_run_id=parent_id,precision_probe_run_id=parent_id,
            qualification='Inherited same-source operation qualification, FP8 NumPy model oracle regression, paired F16 precision probe and native measurements; completes HTTP replay with exact 128 FP8 IDs at every shape')
for row in report['results']:
 row['native_measurement_run_id']=parent_id
 row['serving']=row.pop('engine')
assert len(report['results'])==3 and all(len(r['output_ids'][0])==128 for r in report['results'])
program=Path('build/lfm-moe/codec-model/cuda/program')
rec=Recorder('benchmarks',meta,kind='diagnostic',change='Complete HTTP qualification of the integrated codec FP8 program with owner-thread service construction',parent=parent_id)
passed=False
@contextmanager
def factory():
 with tensor.Device(provider='cuda') as device:
  with LFM2MoE(meta['checkpoint'],program,device,graphs=True) as model:
   with Engine(model,config=EngineConfig(prefill_chunk=4096,prefill_requests=1)) as engine:
    yield Service(engine,model.checkpoint.tokenizer.encode,dict(model=meta['model'],provider='cuda',kv_dtype='fp8'))
try:
 os.environ['AKAS_SHAPE_TIMEOUT']='120';os.environ['AKAS_BENCHMARK_WARMUP']='1';os.environ['AKAS_BENCHMARK_REPEATS']='3'
 print('HTTP_LOADING_AND_CHECKING',flush=True)
 meta['http']=asyncio.run(http_benchmark(factory,report['results'],vocab=128000))
 print('HTTP_PASSED',json.dumps(meta['http']),flush=True)
 for row in report['results']:
  print('RESULT',row['case']['prompt'],'pp',row['summary']['prefill_tokens_per_second']['median'],'tg',row['summary']['decode_tokens_per_second']['median'],'http',json.dumps(row['serving']),flush=True)
 passed=True
except BaseException as error:
 meta['error']=f'{type(error).__name__}: {error}';meta['traceback']=traceback.format_exc();print(meta['traceback'],flush=True)
finally:
 rec.program(program);rec.artifact(__file__,'preflight/http-runner.py')
 write('build/lfm-moe/codec-model/cuda/http-report.json',report)
 rec.update(report,final=True,status='passed' if passed else 'failed');print('RETAINED',rec.id,passed,flush=True)
if not passed:raise SystemExit(1)
