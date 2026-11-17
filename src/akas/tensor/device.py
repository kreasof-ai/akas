"""Shared hardware Vulkan selection for WebGPU execution."""

from contextlib import contextmanager


@contextmanager
def vulkan_device(index=0):
    import tensor

    device = tensor.Device(index, provider="webgpu", max_buffer_size=1024**3)
    adapter = device.info["adapter"]
    if adapter["backend_type"] != "Vulkan" or adapter["adapter_type"] == "CPU":
        raise ValueError("select a hardware Vulkan adapter (WGPU_BACKEND_TYPE=Vulkan)")
    # WGPU treats UINT32_MAX as undefined instead of as the reported limit.
    limit = device._adapter.limits["max-storage-buffers-per-shader-stage"]
    if limit == 0xFFFFFFFF:
        device._adapter._limits["max-storage-buffers-per-shader-stage"] = 32
    with device:
        device.info["adapter_storage_buffers_reported"] = limit
        yield device
