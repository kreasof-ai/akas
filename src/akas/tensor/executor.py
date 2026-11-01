"""Tensor ABI binding, ordered submission and provider-specific capture."""

from tensor.runtime.abi import BoundCall


def bind(device, kernel, arguments):
    if hasattr(kernel, "bind"):
        return kernel.bind(*arguments)
    # Tensor releases before the public binding API retain the ABI-1.3 path.
    values, symbols, launch = kernel._bind(arguments, {}, include_outputs=True)
    return BoundCall(device, kernel.manifest, values, symbols, launch, validated=True)


def submit(device, calls, *, after_launch=None):
    for entry in calls:
        if getattr(entry[1], "executable", None) is not None:
            entry[1].launch()
        else:
            device._launch(entry[0], entry[1])
        if after_launch is not None:
            after_launch(entry)


def capture(device, calls, *, resources):
    if device.info.get("provider", "cuda") == "webgpu":
        return device.prepare_plan([(entry[0], entry[1]) for entry in calls])
    from tensor.providers.cuda.graph import CudaGraph

    return CudaGraph(device, lambda: submit(device, calls), resources=resources)
