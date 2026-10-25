# DSL implementation SHA256: 3d54957f85efc5a2ce332740cd0c9fcba47956dcb03ff8a0fe80ea9e4f7f7876
from akas.kernels.definition import define

def tensor_export():
    return {"kernel": define({'contract': 'reference', 'kind': 'ffn', 'parameters': {'ggml': True, 'half_accum': True, 'k': 2048, 'o': 10752, 'optimized': True, 'r': 512, 'type': 1}, 'phase': 'prefill'}, {'half_k': 32, 'half_layout': 0, 'half_native': False, 'half_pad': 1}, {'provider': 'webgpu', 'target': 'webgpu-portable-v1'}, {'axes': ['half_layout', 'half_pad', 'half_native', 'half_k'], 'contract': 'reference', 'defaults': {'half_k': 16, 'half_layout': 0, 'half_native': True, 'half_pad': 1}, 'factory': 'half_prefill', 'identifier': 'linear.webgpu.short_half', 'kind': 'ffn', 'module': 'akas.ops.linear.webgpu', 'phase': 'forward', 'provider': 'webgpu', 'semantic_keys': ['r', 'k', 'o', 'type', 'ggml', 'half_accum'], 'targets': []})}
