from pathlib import Path
import json,xml.etree.ElementTree as ET
from akas.benchmarking.history import Recorder,write
w=Path('build/spark-parity');h=Path('benchmarks')
suites=list(ET.parse(w/'qualification-compact.xml').getroot().iter('testsuite'))
assert suites and all(int(s.get('failures','0'))==int(s.get('errors','0'))==0 for s in suites)
counts={k:sum(int(s.get(k,'0')) for s in suites) for k in ('tests','failures','errors','skipped')}
meta=dict(runtime='akas',purpose='model-qualification',model='XHToken/Spark-X2.6-4B',
 mode='ar',weight_format='q4_k_m',provider='cuda',target='sm_89',kv_dtype='float16',
 qualification_formats=['mixed-gguf','bf16','fp8-channel'], independent_small_model_reference=True,
 tests=counts,persistent_dequantized_weight_bytes=0,
 command='AKAS_SPARK_CUDA=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TENSOR_NVRTC_HOME=build/nvrtc-12.9 python -m pytest tests/test_spark_performance.py tests/test_phase_memory.py tests/test_spark.py',
 scope='Independent packed projections, fused exact GELU, FP16 activation staging, residual RMS, compact RoPE, fused QKV/cache, compact sliding rings with staged current K/V and post-attention suffix commits, NaN padding, byte-preserving host/device weight arenas, scratch ownership, and complete BF16/FP8/mixed-GGUF equations across cache wrap, physical slot reorder and reuse',
 prior_real_fp8_checkpoint_smoke_run='20261026111621594068-f8c4e17b')
r=Recorder(h,meta,kind='verification',change='Qualify compact quantized Spark prefill and allocation ownership')
for name in ('qualification-compact.log','qualification-compact.xml','regression-compact.log'):
 r.artifact(w/name,'validation/'+name)
for name in ('test_spark.py','test_spark_performance.py','test_spark_weight_storage.py','test_phase_memory.py'):
 r.artifact(Path('tests')/name,'validation/'+name)
r.artifact('tests/support/spark.py','validation/spark_reference.py')
r.artifact(w/'qualify.py','validation/qualify.py')
x=r.update(dict(schema='akas.benchmark.v1',metadata=meta,results=[]),final=True,status='passed')
assert x['status']=='passed'
write(w/'qualification.json',dict(run_id=r.id,tests=counts))
print(json.dumps(dict(run_id=r.id,tests=counts)),flush=True)
