# DSL implementation SHA256: 539b6902c12184af2eeef84921e23cf76f593b9572f2748153c00228380a6e79
from akas.kernels.definition import define

def tensor_export():
    return {"kernel": define({'contract': 'reference', 'kind': 'ffn', 'parameters': {'ggml': True, 'half_accum': True, 'k': 2048, 'o': 10752, 'optimized': True, 'r': 512, 'type': 1}, 'phase': 'prefill'}, {'half_layout': 0, 'half_pad': 1}, {'provider': 'webgpu', 'target': 'webgpu-portable-v1'}, {'axes': ['half_layout', 'half_pad'], 'contract': 'reference', 'defaults': {'half_layout': 0, 'half_pad': 1}, 'factory': 'half_prefill', 'identifier': 'linear.webgpu.short_half', 'kind': 'ffn', 'module': 'akas.ops.linear.webgpu', 'phase': 'forward', 'provider': 'webgpu', 'semantic_keys': ['r', 'k', 'o', 'type', 'ggml', 'half_accum'], 'targets': []})}
