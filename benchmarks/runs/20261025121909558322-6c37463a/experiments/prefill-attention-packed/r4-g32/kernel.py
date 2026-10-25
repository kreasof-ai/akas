# DSL implementation SHA256: c1f54f3e31319fcfbcb5bacd8bb60f09f98198c35f1d9bbb2a50c014cf503259
from akas.kernels.definition import define

def tensor_export():
    return {"kernel": define({'contract': 'reference', 'kind': 'attention', 'parameters': {'block_k': 32, 'block_q': 4, 'capacity': 131072, 'chunk': 512, 'd': 64, 'dot_width': 4, 'h': 32, 'key_pad': 0, 'kh': 8, 'kv_dtype': 'float16', 'optimized': True, 'packed_shared': True, 'provider': 'webgpu', 'r': 512, 'shared_dtype': 'float16', 'unit_scales': True}, 'phase': 'forward'}, {}, {'provider': 'webgpu', 'target': 'webgpu-portable-v1'}, {'axes': ['block_q', 'block_k', 'key_pad', 'shared_dtype', 'dot_width', 'packed_shared'], 'contract': 'reference', 'defaults': {}, 'factory': 'tiled_prefill', 'identifier': 'attention.webgpu.tiled_prefill', 'kind': 'attention', 'module': 'akas.ops.attention.webgpu', 'phase': 'forward', 'provider': 'webgpu', 'semantic_keys': None, 'targets': []})}
