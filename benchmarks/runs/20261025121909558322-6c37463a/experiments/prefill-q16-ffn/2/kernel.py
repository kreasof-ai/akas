# DSL implementation SHA256: df726e6b3522aaaf841ab2da7ec65321b597837abe4b5bad3625803c6fc06c74
from akas.kernels.definition import define

def tensor_export():
    return {"kernel": define({'contract': 'reference', 'kind': 'ffn', 'parameters': {'ggml': True, 'integer': {'fixed_residual': False}, 'integer_k': 32, 'integer_layout': 4, 'k': 2048, 'o': 10752, 'optimized': True, 'q16': True, 'r': 512, 'type': 2}, 'phase': 'forward'}, {}, {'provider': 'webgpu', 'target': 'webgpu-portable-v1'}, {'axes': ['integer_layout', 'integer_k'], 'contract': 'reference', 'defaults': {'integer_k': 32, 'integer_layout': 0}, 'factory': 'integer_prefill', 'identifier': 'linear.webgpu.two_component', 'kind': 'ffn', 'module': 'akas.ops.linear.webgpu', 'phase': 'forward', 'provider': 'webgpu', 'semantic_keys': ['r', 'k', 'o', 'type', 'ggml', 'q16'], 'targets': []})}
