# DSL implementation SHA256: 7dc0250e5ae14cef7efdeca16d2dd3462048835f2fb1a81ae0728ff8e8e9c80e
from akas.kernels.definition import define

def tensor_export():
    return {"kernel": define({'contract': 'reference', 'kind': 'attention_adaptive_partial', 'parameters': {'capacity': 131072, 'd': 64, 'dot_chains': 1, 'granule': 128, 'h': 32, 'key_rows': 64, 'kh': 8, 'kv_dtype': 'float16', 'optimized': True, 'provider': 'webgpu', 'r': 1, 'splits': 16, 'unit_scales': True}, 'phase': 'forward'}, {}, {'provider': 'webgpu', 'target': 'webgpu-portable-v1'}, {'axes': ['key_rows', 'granule', 'dot_chains'], 'contract': 'reference', 'defaults': {'dot_chains': 1, 'granule': 128, 'key_rows': 32}, 'factory': 'packed_channel_partial', 'identifier': 'attention.webgpu.packed_decode', 'kind': 'attention_adaptive_partial', 'module': 'akas.ops.attention.webgpu', 'phase': 'forward', 'provider': 'webgpu', 'semantic_keys': ['r', 'h', 'kh', 'd', 'capacity', 'splits', 'kv_dtype'], 'targets': []})}
