# DSL implementation SHA256: 4c0e0442718a1251ad76b15c0c5f1fe17822a69e8f9407250051d0e7cbbdfbfc
from akas.kernels.definition import define

def tensor_export():
    return {"kernel": define({'contract': 'reference', 'kind': 'linear', 'parameters': {'ggml': True, 'integer': {'fixed_residual': False}, 'integer_k': 128, 'integer_layout': 1, 'k': 10752, 'o': 2048, 'optimized': True, 'q16': True, 'r': 512, 'type': 2}, 'phase': 'forward'}, {}, {'provider': 'webgpu', 'target': 'webgpu-portable-v1'}, {'axes': ['integer_layout', 'integer_k'], 'contract': 'reference', 'defaults': {'integer_k': 32, 'integer_layout': 0}, 'factory': 'integer_prefill', 'identifier': 'linear.webgpu.two_component', 'kind': 'linear', 'module': 'akas.ops.linear.webgpu', 'phase': 'forward', 'provider': 'webgpu', 'semantic_keys': ['r', 'k', 'o', 'type', 'ggml', 'q16'], 'targets': []})}
