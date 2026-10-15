"""Verify prepared submission preserves the imperative model's tokens/logits."""
import json
from pathlib import Path
import numpy as np
import tensor
from akas.benchmarking.history import Recorder
from akas.benchmarking.session import Session
from akas.models.lfm2.model import LFM2

script = Path(__file__).resolve()
metadata = dict(runtime='akas', model='LiquidAI/LFM2.5-230M', provider='webgpu',
                target='webgpu-portable-v1', purpose='validation', mode='ar',
                protocol='execution-equivalence-v1', weight_format='q4_k_m',
                kv_dtype='float16', seed=17)
recorder = Recorder('benchmarks', metadata,
                    parent='20261015165512907548-25b0d0de', kind='validation',
                    change='Compare identical tokens and logits across imperative and native prepared WebGPU execution')
recorder.artifact(script, 'validation.py')
report = dict(schema='akas.benchmark.v1', metadata=metadata, results=[])
try:
    outputs, logits = [], []
    with tensor.Device(0, provider='webgpu') as device:
        metadata['device'] = dict(device.info)
        for graphs in (False, True):
            with LFM2('build/checkpoints/lfm2.5-230m/q4_k_m.gguf',
                      'build/model-smoke/banks/lfm-gguf/webgpu-float16-program',
                      device, graphs=graphs) as model:
                with Session(model, batch=1, chunk=32) as session:
                    session.reset()
                    prompt = np.random.default_rng(17).integers(0, session.config.vocab, 32, dtype='int32')
                    tokens = [list(session.prefill([prompt]))]
                    features = [session.read_feature('logits')]
                    for _ in range(7):
                        tokens.append(list(session.decode(1)))
                        features.append(session.read_feature('logits'))
                    outputs.append(np.asarray(tokens))
                    logits.append(np.asarray(features))
        device.synchronize()
    finite = all(np.isfinite(value).all() for value in logits)
    same_tokens = np.array_equal(*outputs)
    same_logits = np.array_equal(*logits)
    correct = finite and same_tokens and same_logits
    evidence = dict(correct=correct, finite=finite, identical_tokens=same_tokens,
                    identical_logits=same_logits, max_absolute_error=float(np.max(np.abs(logits[0]-logits[1]))),
                    output_token_ids=outputs[0].tolist())
    report['results'].append(dict(case=dict(batch=1,prompt=32,generated=8,depth=0,mixed=False),
                                  status='ok' if correct else 'error', correctness=evidence))
    recorder.program('build/model-smoke/banks/lfm-gguf/webgpu-float16-program')
    print(json.dumps(evidence))
    if not correct:
        raise ValueError('prepared execution differs from imperative execution')
finally:
    recorder.update(report, final=True)
    print('Run:', recorder.id)
