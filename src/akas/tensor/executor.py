"""Tensor ABI binding, ordered submission and provider-specific capture."""

from tensor.runtime.abi import BoundCall


def bind(device, kernel, arguments):
    values, symbols, launch = kernel._bind(arguments, {}, include_outputs=True)
    return BoundCall(device, kernel.manifest, values, symbols, launch, validated=True)


def submit(device, calls, *, after_launch=None):
    for entry in calls:
        device._launch(entry[0], entry[1])
        if after_launch is not None:
            after_launch(entry)


def capture(device, calls, *, resources):
    if device.info.get("provider", "cuda") == "webgpu":
        return device.prepare_plan(calls)
    from tensor.providers.cuda.graph import CudaGraph

    return CudaGraph(device, lambda: submit(device, calls), resources=resources)
