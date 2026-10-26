from pathlib import Path
import json,xml.etree.ElementTree as ET
from akas.benchmarking.history import Recorder,write
w=Path('build/spark-parity');h=Path('benchmarks')
suites=list(ET.parse(w/'qualification-corrected.xml').getroot().iter('testsuite'))
assert suites and all(int(s.get('errors','0'))==0 for s in suites)
current=ET.parse(w/'qualification-corrected.xml').getroot()
baseline=ET.parse(w/'baseline-legacy.xml').getroot()
failed={c.get('name') for c in current.iter('testcase') if c.find('failure') is not None}
expected={c.get('name') for c in baseline.iter('testcase') if c.find('failure') is not None}
assert failed==expected=={'test_native_state_algorithms_preserve_math_with_reordered_pool[attention]', 'test_mapped_restore_flattens_view_indices_using_view_shape[view]'}
assert json.loads((w/'pruning-parity.json').read_text())['bitwise_equal']

counts={k:sum(int(s.get(k,'0')) for s in suites) for k in ('tests','failures','errors','skipped')}
meta=dict(runtime='akas',purpose='model-qualification',model='XHToken/Spark-X2.6-4B',
 mode='ar',weight_format='q4_k_m',provider='cuda',target='sm_89',kv_dtype='float16',
 qualification_formats=['mixed-gguf','bf16','fp8-channel'], independent_small_model_reference=True,
 tests=counts,known_baseline_failures=sorted(failed),passed_checks=counts['tests']-counts['failures']-counts['skipped'],persistent_dequantized_weight_bytes=0,real_checkpoint_pruned_vs_full_bitwise_equal=True,
 command='AKAS_SPARK_CUDA=1 AKAS_LFM_REFERENCE_CUDA=1 TENSOR_QWEN_CUDA=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TENSOR_NVRTC_HOME=build/nvrtc-12.9 python -m pytest tests/test_spark_performance.py tests/test_phase_memory.py tests/test_spark.py tests/test_state_prefill.py tests/test_state_mapping.py',
 scope='Two unchanged legacy state-mapping tests reference removed entry points and fail identically on the pre-fix checkout; retained as exclusions. Independent packed projections, fused exact GELU, FP16 activation staging, residual RMS, compact RoPE, fused QKV/cache, compact sliding rings with staged current K/V and post-attention suffix commits, NaN padding, byte-preserving host/device weight arenas, scratch ownership, and complete BF16/FP8/four-layer mixed-GGUF equations across cache wrap, physical slot reorder and reuse',
 prior_real_fp8_checkpoint_smoke_run='20261026111621594068-f8c4e17b')
r=Recorder(h,meta,kind='verification',change='Qualify compact quantized Spark prefill and allocation ownership')
for name in ('qualification-corrected.log','qualification-corrected.xml','regression-corrected.log','baseline-legacy.log','baseline-legacy.xml','pruning-parity.py','pruning-parity.json','pruning-parity.log'):
 r.artifact(w/name,'validation/'+name)
for name in ('test_spark.py','test_spark_performance.py','test_spark_weight_storage.py','test_phase_memory.py','test_state_prefill.py','test_state_mapping.py'):
 r.artifact(Path('tests')/name,'validation/'+name)
r.artifact('tests/support/spark.py','validation/spark_reference.py')
r.artifact(w/'qualify.py','validation/qualify.py')
x=r.update(dict(schema='akas.benchmark.v1',metadata=meta,results=[]),final=True,status='passed')
assert x['status']=='passed'
write(w/'qualification.json',dict(run_id=r.id,tests=counts))
print(json.dumps(dict(run_id=r.id,tests=counts)),flush=True)
