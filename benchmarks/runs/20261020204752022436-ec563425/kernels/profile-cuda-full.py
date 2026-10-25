"""Attribute the qualified full CUDA program without mixing profiling with rates."""
import json
from pathlib import Path
import traceback

import tensor
from akas.benchmarking.history import Recorder, write
from akas.benchmarking.session import Session
from akas.models.lfm2.model import LFM2MoE
from akas.optimization.evaluation import prompts
from cuda_profile import Profile, summarize

program = Path('build/lfm-moe/cuda-full/cuda/program')
parent = json.loads(Path('build/lfm-moe/cuda-full/cuda/run-id.json').read_text())['run_id']
reference = json.loads((Path('benchmarks/runs') / parent / 'run.json').read_text())
if reference['status'] != 'passed':
    raise ValueError('Full CUDA native/Engine/HTTP benchmark must pass before profiling')
root = Path('build/lfm-moe/cuda-full-profile')
root.mkdir(exist_ok=True)
meta = dict(reference['metadata'])
meta.update(purpose='kernel', protocol='ordered-forward-profile-v1',
    reference_run_id=parent, requested_prefixes=[32, 8192, 130944],
    qualification='Diagnostic ordered CUDA events perturb execution. Prefill attribution describes the final chunk at each context, not total prompt throughput. Decode uses real successive token states. Every 128-token sequence must match the qualified native/Engine/HTTP run. Use that run for throughput claims.')
for field in ('http', 'same_precision_logit_comparison', 'precision_comparison', 'compilation_resources'):
    meta.pop(field, None)
rec = Recorder('benchmarks', meta, kind='diagnostic', parent=parent,
    change='Attribute source-integrated fused decode and staged prefill after the full CUDA context sweep')
report = dict(schema='akas.benchmark.v1', metadata=meta, results=[])
passed = False
try:
    with tensor.Device(provider='cuda') as device:
        with LFM2MoE(meta['checkpoint'], program, device, graphs=True) as model:
            for prefix in (32, 8192, 130944):
                chunk = 32 if prefix == 32 else 4096
                frame = model.batches.select(1, chunk)
                profile = Profile(frame.forward)
                original = frame.forward.run
                samples = []

                def run_profiled():
                    profile.graph.launch()
                    samples.append(profile.read())

                print('PROFILE_PREFILL_FINAL_CHUNK', prefix, flush=True)
                try:
                    with Session(model, batch=1, chunk=4096, timeout=120) as session:
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
                result.update(prefix=prefix, chunk=chunk, samples=len(samples), phase='prefill-final-chunk')
                meta.setdefault('profiles', []).append(result)
                print('PROFILE', json.dumps({k: v for k, v in result.items() if k != 'calls'}), flush=True)
                write(root / 'report.json', report)
                with Session(model, batch=1, chunk=4096, timeout=120) as session:
                    session.reset()
                    ids = list(session.prefill(prompts(model.config.vocab, 1, prefix)))
                    decode = model.batches.select(1, 1).forward
                    profile = Profile(decode)
                    original = decode.graph
                    decode.graph = profile.graph
                    samples = []
                    try:
                        for _ in range(127):
                            ids.extend(session.decode(1))
                            samples.append(profile.read())
                    finally:
                        decode.graph = original
                        profile.close()
                expected = next(r for r in reference['results'] if r['case']['prompt'] == prefix)['output_ids']
                if [ids] != expected:
                    raise ValueError(f'Ordered graph events change 128 model output IDs at {prefix}')
                result = summarize(profile, samples[2:])
                result.update(prefix=prefix, samples=len(samples) - 2, phase='decode', identical_output=True)
                meta['profiles'].append(result)
                print('PROFILE', json.dumps({k: v for k, v in result.items() if k != 'calls'}), flush=True)
                write(root / 'report.json', report)
    passed = True
except BaseException:
    meta['traceback'] = traceback.format_exc()
    print(meta['traceback'], flush=True)
finally:
    rec.program(program)
    rec.artifact(__file__, 'kernels/profile-cuda-full.py')
    rec.artifact('build/lfm-moe/cuda_profile.py', 'kernels/cuda_profile.py')
    write(root / 'report.json', report)
    rec.update(report, final=True, status='passed' if passed else 'failed')
    print('RETAINED', rec.id, passed, flush=True)
if not passed:
    raise SystemExit(1)
