"""Diagnostic standard WebGPU pass timestamps; no provider source is modified."""
from contextlib import contextmanager
import numpy as np


@contextmanager
def timestamp_device():
    import tensor

    device = tensor.Device(provider='webgpu')
    adapter = device._adapter
    if 'timestamp-query' not in adapter.features:
        raise ValueError('WebGPU adapter does not expose timestamp-query')
    request = adapter.request_device_sync
    sentinel = object()
    original = vars(adapter).get('request_device_sync', sentinel)

    def with_timestamps(**kwargs):
        kwargs['required_features'] = [*kwargs.get('required_features', []), 'timestamp-query']
        return request(**kwargs)

    adapter.request_device_sync = with_timestamps
    try:
        with device as opened:
            yield opened
    finally:
        if original is sentinel:
            del adapter.request_device_sync
        else:
            adapter.request_device_sync = original


class Profile:
    """One timestamped pass per ordered dispatch; intentionally diagnostic only."""
    def __init__(self, forward):
        import wgpu

        self.device = forward.device
        self.plan = self.device.prepare_plan([(e[0], e[1]) for e in forward.plan])
        self.count = 2 * len(self.plan.nodes)
        self.query = self.device._gpu.create_query_set(type='timestamp', count=self.count)
        self.resolved = self.device._gpu.create_buffer(size=(self.count * 8 + 255) // 256 * 256,
            usage=wgpu.BufferUsage.QUERY_RESOLVE | wgpu.BufferUsage.COPY_SRC)
        self.calls = [dict(kind=c.kind, label=e[2], parameters=c.parameters)
                      for e, c in zip(forward.plan, forward.definition.calls)]
        assert len(self.calls) == len(self.plan.nodes)
        self.closed = False

    def launch(self):
        if self.closed:
            raise RuntimeError('Timestamp profile is closed')
        gpu = self.device._gpu
        encoder = gpu.create_command_encoder()
        dimension = gpu.limits['max-compute-workgroups-per-dimension']
        for index, (pipeline, group, grid) in enumerate(self.plan.nodes):
            compute = encoder.begin_compute_pass(timestamp_writes=dict(query_set=self.query,
                beginning_of_pass_write_index=index * 2, end_of_pass_write_index=index * 2 + 1))
            compute.set_pipeline(pipeline)
            compute.set_bind_group(0, group, [])
            width = min(grid[0], dimension)
            compute.dispatch_workgroups(width, grid[1], (grid[0] + width - 1) // width)
            compute.end()
        encoder.resolve_query_set(self.query, 0, self.count, self.resolved, 0)
        gpu.queue.submit([encoder.finish()])

    def read(self):
        self.device.synchronize()
        values = np.frombuffer(self.device._gpu.queue.read_buffer(self.resolved, 0, self.count * 8), dtype='<u8')
        pairs = values.reshape(-1, 2)
        if np.any(pairs[:, 1] < pairs[:, 0]):
            raise ValueError('WebGPU timestamps are not ordered')
        return ((pairs[:, 1] - pairs[:, 0]).astype('float64') / 1e6).tolist()

    def close(self):
        if self.closed:
            return
        self.device.synchronize()
        self.query.destroy()
        self.resolved.destroy()
        self.plan.close()
        self.closed = True
