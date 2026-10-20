"""Measure actual WebGPU operation costs after the full CUDA sweep has passed."""
import json
from pathlib import Path
import traceback

from akas.benchmarking.history import Recorder, write
from akas.benchmarking.session import Session
from akas.models.lfm2.model import LFM2MoE
from akas.optimization.evaluation import prompts
from cuda_profile import summarize
from wgpu_profile import Profile, timestamp_device
from akas.artifacts.read import read_program
from akas.graph import Graph
from akas.optimization.planning import estimate

cuda_id = json.loads(Path('build/lfm-moe/cuda-full/cuda/run-id.json').read_text())['run_id']
if json.loads((Path('benchmarks/runs') / cuda_id / 'run.json').read_text())['status'] != 'passed':
    raise ValueError('Complete and validate the full CUDA sweep before WebGPU profiling')
program = Path('build/lfm-moe/manual-v2-matched/webgpu/program')
reference_path = program.parent / 'report.json'
reference = json.loads(reference_path.read_text())
needed = max(estimate(Graph.from_metadata(json.loads((program / p['path']).read_text())['forward']))['largest_buffer_bytes']
             for p in read_program(program)['phases'])
meta = dict(reference['metadata'])
meta.update(purpose='kernel', protocol='ordered-webgpu-pass-profile-v1', cuda_run_id=cuda_id,
    qualification='Diagnostic per-dispatch standard WebGPU timestamps add pass boundaries and perturb execution. Prefill attribution describes the final chunk only; decode is successive real-token state. Native/HTTP throughput must be measured separately. Every 128-token sequence must match the retained F16 WebGPU reference.')
meta['max_buffer_size'] = needed
for field in ('compilation_resources', 'cpu_peak_rss_bytes'):
    meta.pop(field, None)
root = Path('build/lfm-moe/webgpu-current-profile')
root.mkdir(exist_ok=True)
rec = Recorder('benchmarks', meta, kind='diagnostic',
    change='Identify measured WebGPU kernel bottlenecks before its performance catch-up work')
report = dict(schema='akas.benchmark.v1', metadata=meta, results=[])
passed = False
try:
    with timestamp_device(max_buffer_size=needed) as device:
        meta['timestamp_query'] = True
        with LFM2MoE(meta['checkpoint'], program, device, graphs=True) as model:
            for prefix in (32, 8192):
                chunk = 32 if prefix == 32 else 512
                frame = model.batches.select(1, chunk)
                profile = Profile(frame.forward)
                original = frame.forward.run
                samples = []

                def run_profiled():
                    profile.launch()
                    samples.append(profile.read())

                print('PROFILE_PREFILL_FINAL_CHUNK', prefix, flush=True)
                try:
                    with Session(model, batch=1, chunk=512, timeout=120) as session:
                        session.reset()
                        session.prefill(prompts(model.config.vocab, 1, prefix), read=False)
                        frame.forward.run = run_profiled
                        for _ in range(2):
                            session.reset()
                            session.prefill(prompts(model.config.vocab, 1, prefix), read=False)
                finally:
                    frame.forward.run = original
                    profile.close()
                result = summarize(profile, samples)
                result.update(prefix=prefix, chunk=chunk, phase='prefill-final-chunk', samples=len(samples))
                meta.setdefault('profiles', []).append(result)
                print('PROFILE', json.dumps({k: v for k, v in result.items() if k != 'calls'}), flush=True)
                write(root / 'report.json', report)
                with Session(model, batch=1, chunk=512, timeout=120) as session:
                    session.reset()
                    ids = list(session.prefill(prompts(model.config.vocab, 1, prefix)))
                    frame = model.batches.select(1, 1)
                    profile = Profile(frame.forward)
                    original = frame.forward.graph
                    frame.forward.graph = profile
                    samples = []
                    try:
                        for _ in range(127):
                            ids.extend(session.decode(1))
                            samples.append(profile.read())
                    finally:
                        frame.forward.graph = original
                        profile.close()
                expected = next(r for r in reference['results'] if r['case']['prompt'] == prefix)['output_ids']
                if [ids] != expected:
                    raise ValueError(f'WebGPU timestamps change qualified token IDs at {prefix}')
                result = summarize(profile, samples[2:])
                result.update(prefix=prefix, phase='decode', samples=len(samples) - 2, identical_output=True)
                meta['profiles'].append(result)
                print('PROFILE', json.dumps({k: v for k, v in result.items() if k != 'calls'}), flush=True)
                write(root / 'report.json', report)
    passed = True
except BaseException:
    meta['traceback'] = traceback.format_exc()
    print(meta['traceback'], flush=True)
finally:
    rec.program(program)
    rec.artifact(__file__, 'kernels/profile-webgpu-current.py')
    rec.artifact('build/lfm-moe/wgpu_profile.py', 'kernels/wgpu_profile.py')
    rec.artifact('build/lfm-moe/cuda_profile.py', 'kernels/cuda_profile.py')
    rec.artifact('build/lfm-moe/check-wgpu-timestamps.py', 'validation/check-wgpu-timestamps.py')
    rec.artifact('build/lfm-moe/wgpu-timestamps-check.log', 'validation/wgpu-timestamps-check.log')
    rec.artifact(reference_path, 'reference/webgpu-report.json')
    write(root / 'report.json', report)
    rec.update(report, final=True, status='passed' if passed else 'failed')
    print('RETAINED', rec.id, passed, flush=True)
if not passed:
    raise SystemExit(1)
