# DSL implementation SHA256: 96a8cf2ec74233488009b5c2db958f1ca98919b7c353036a136ec204d84e21a6
from akas.kernels.definition import define

def tensor_export():
    return {"kernel": define({'contract': 'reference', 'kind': 'ffn', 'parameters': {'ggml': True, 'half_accum': True, 'k': 256, 'o': 128, 'r': 32, 'type': 1}, 'phase': 'forward'}, {}, {'provider': 'webgpu', 'target': 'webgpu-portable-v1'}, {'axes': ['half_layout', 'half_pad', 'half_native'], 'contract': 'reference', 'defaults': {'half_layout': 0, 'half_native': True, 'half_pad': 1}, 'factory': 'half_prefill', 'identifier': 'linear.webgpu.short_half', 'kind': 'ffn', 'module': 'akas.ops.linear.webgpu', 'phase': 'forward', 'provider': 'webgpu', 'semantic_keys': ['r', 'k', 'o', 'type', 'ggml', 'half_accum'], 'targets': []})}
