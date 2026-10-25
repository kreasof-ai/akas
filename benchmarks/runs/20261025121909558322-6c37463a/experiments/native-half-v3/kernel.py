# DSL implementation SHA256: a4bffc02fbea091894a42cfce27dd9ed78ac57032adcbf9efcb3909bde00ab8e
from akas.kernels.definition import define

def tensor_export():
    return {"kernel": define({'contract': 'reference', 'kind': 'ffn', 'parameters': {'ggml': True, 'half_accum': True, 'k': 256, 'o': 128, 'r': 32, 'type': 1}, 'phase': 'forward'}, {}, {'provider': 'webgpu', 'target': 'webgpu-portable-v1'}, {'axes': ['half_layout', 'half_pad', 'half_native'], 'contract': 'reference', 'defaults': {'half_layout': 0, 'half_native': True, 'half_pad': 1}, 'factory': 'half_prefill', 'identifier': 'linear.webgpu.short_half', 'kind': 'ffn', 'module': 'akas.ops.linear.webgpu', 'phase': 'forward', 'provider': 'webgpu', 'semantic_keys': ['r', 'k', 'o', 'type', 'ggml', 'half_accum'], 'targets': []})}
