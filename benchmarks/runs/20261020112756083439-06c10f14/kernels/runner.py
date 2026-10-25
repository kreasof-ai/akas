"""Bounded whole-model CUDA graph profiling; preserve call order and real routing."""
import collections
import ctypes as c
import json
import statistics
import time
from pathlib import Path

import numpy as np
import tensor
from tensor.providers.cuda.graph import CudaGraph
from akas.models.lfm2.model import LFM2MoE
from akas.benchmarking.session import Session
from akas.benchmarking.history import Recorder, write

root = Path('build/lfm-moe/whole-model-profile')
root.mkdir(parents=True, exist_ok=True)
metadata = dict(runtime='akas', purpose='kernel', protocol='kernel-latency-v1',
                provider='cuda', model='akas.models.lfm2.model:LFM2MoE',
                checkpoint='build/checkpoints/lfm2.5-24b-a2b/q4_0.gguf',
                checkpoint_sha256='d1c03f73ecaf61017d70419eec9e5dc0b05d50ae26c23d522829b934fbabf8a5',
                device=dict(name='NVIDIA L4', provider='cuda'),
                qualification='Profiling only; captured event nodes perturb timings; real whole-model greedy decode and streaming weights')
rec = Recorder('benchmarks', metadata, kind='diagnostic', change='Profile ordered full-model execution before further algorithm changes')
report = dict(schema='akas.benchmark.v1', metadata=metadata, results=[])

class Profile:
    def __init__(self, forward):
        self.device = device = forward.device
        self.events = [c.c_void_p() for _ in range(len(forward.plan) + 1)]
        device.driver.lib.cuEventRecordWithFlags.argtypes = [c.c_void_p, c.c_void_p, c.c_uint]
        device.driver.lib.cuEventRecordWithFlags.restype = c.c_int
        for e in self.events:
            device.driver.call('cuEventCreate', c.byref(e), 0)
        self.calls = []
        for entry, call in zip(forward.plan, forward.definition.calls):
            kernel = entry[0]
            self.calls.append(dict(kind=call.kind, label=entry[2], parameters=call.parameters,
                                   implementation=kernel.manifest.get('implementation'),
                                   name=kernel.manifest.get('name')))
        def submit():
            device.driver.call('cuEventRecordWithFlags', self.events[0], device.stream, 1)
            for entry, event in zip(forward.plan, self.events[1:]):
                device._launch(entry[0], entry[1])
                device.driver.call('cuEventRecordWithFlags', event, device.stream, 1)
        self.graph = CudaGraph(device, submit, resources=tuple(forward.buffers.values()) + tuple(forward.kernels.values()))

    def read(self):
        self.device.synchronize()
        times = []
        for a, b in zip(self.events, self.events[1:]):
            value = c.c_float()
            self.device.driver.call('cuEventElapsedTime', c.byref(value), a, b)
            times.append(value.value)
        return times

    def close(self):
        self.graph.close()
        for e in self.events:
            self.device.driver.call('cuEventDestroy_v2', e)

def summarize(profile, samples):
    medians = np.median(np.asarray(samples), axis=0)
    grouped = collections.defaultdict(float)
    records = []
    for call, milliseconds in zip(profile.calls, medians):
        grouped[call['kind']] += float(milliseconds)
        records.append(dict(**call, milliseconds=float(milliseconds)))
    return dict(total_ms=float(medians.sum()), groups_ms=dict(sorted(grouped.items(), key=lambda x:-x[1])), calls=records)

passed = False
try:
    with tensor.Device(provider='cuda') as device:
        with LFM2MoE(metadata['checkpoint'], 'build/lfm-moe/manual-v2/cuda/program', device, graphs=True) as executor:
            rng = np.random.default_rng(17)
            prompt = rng.integers(0, executor.config['vocab_size'] if isinstance(executor.config, dict) and 'vocab_size' in executor.config else 128000, size=8192).tolist()
            with Session(executor, batch=1, chunk=512) as session:
                session.reset()
                session.prefill([prompt], read=False)
                session.decode(4, read=False)
                device.synchronize()
                start=time.perf_counter(); session.decode(16, read=False);device.synchronize()
                baseline_before=(time.perf_counter()-start)/16*1000
                frame=executor.last_frame
                profile=Profile(frame.forward)
                original=frame.forward.graph
                frame.forward.graph=profile.graph
                samples=[]
                try:
                    for i in range(32):
                        session.decode(1, read=False)
                        samples.append(profile.read())
                finally:
                    frame.forward.graph=original
                    profile.close()
                start=time.perf_counter();session.decode(16,read=False);device.synchronize()
                baseline_after=(time.perf_counter()-start)/16*1000
                result=summarize(profile,samples)
                result.update(baseline_before_ms=baseline_before,baseline_after_ms=baseline_after,samples=32,prefix=8192,batch=1)
                metadata['decode_profile']=result
                write(root/'decode.json', result)
                print('DECODE', json.dumps({k:v for k,v in result.items() if k!='calls'}),flush=True)
            # Profile actual prefill chunks, with natural layer/weight ordering.
            frame=executor.batches.select(1,512)
            profile=Profile(frame.forward);original=frame.forward.graph
            original_run=frame.forward.run
            samples=[]
            def run_profiled():
                profile.graph.launch();samples.append(profile.read())
            frame.forward.run=run_profiled
            try:
                with Session(executor,batch=1,chunk=512) as session:
                    session.reset();session.prefill([prompt],read=False)
            finally:
                frame.forward.run=original_run
                profile.close()
            result=summarize(profile,samples)
            result.update(samples=len(samples),prefix=8192,batch=1,chunk=512)
            metadata['prefill_profile']=result
            write(root/'prefill.json',result)
            print('PREFILL',json.dumps({k:v for k,v in result.items() if k!='calls'}),flush=True)
            followup=Path('build/lfm-moe/profile-followup.py')
            if followup.exists():
                exec(compile(followup.read_text(), str(followup), 'exec'))
    passed=True
finally:
    rec.artifact(__file__, 'kernels/runner.py')
    for name in ('decode.json','prefill.json'):
        if (root/name).exists():rec.artifact(root/name,'kernels/'+name)
    rec.update(report, final=True,status='passed' if passed else 'failed')
    print('RETAINED',rec.id,passed,flush=True)
