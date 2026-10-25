import collections
import ctypes as c
import numpy as np
from tensor.providers.cuda.graph import CudaGraph
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
