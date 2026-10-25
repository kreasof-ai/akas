# DSL implementation SHA256: f30a9ff18d3f864ededcaa2d50ffdbea737165e9b4a8748a0d81043fe7bdeaed
from akas.kernels.definition import define

def tensor_export():
    return {"kernel": define({'contract': 'reference', 'kind': 'linear', 'parameters': {'ggml': True, 'half_accum': True, 'k': 2048, 'o': 512, 'optimized': True, 'r': 512, 'type': 2}, 'phase': 'prefill'}, {'half_layout': 0, 'half_pad': 0}, {'provider': 'webgpu', 'target': 'webgpu-portable-v1'}, {'axes': ['half_layout', 'half_pad'], 'contract': 'reference', 'defaults': {'half_layout': 0, 'half_pad': 1}, 'factory': 'half_prefill', 'identifier': 'linear.webgpu.short_half', 'kind': 'linear', 'module': 'akas.ops.linear.webgpu', 'phase': 'forward', 'provider': 'webgpu', 'semantic_keys': ['r', 'k', 'o', 'type', 'ggml', 'half_accum'], 'targets': []})}
