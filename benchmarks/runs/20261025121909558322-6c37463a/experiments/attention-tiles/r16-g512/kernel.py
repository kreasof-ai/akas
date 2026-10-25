# DSL implementation SHA256: 83e75f2726f54dbcd37f676348dfae1657d647f2e380bfcdb5ea60f433562676
from akas.kernels.definition import define

def tensor_export():
    return {"kernel": define({'contract': 'reference', 'kind': 'attention_adaptive_partial', 'parameters': {'capacity': 131072, 'd': 64, 'granule': 512, 'h': 32, 'key_rows': 16, 'kh': 8, 'kv_dtype': 'float16', 'optimized': True, 'provider': 'webgpu', 'r': 1, 'splits': 64, 'unit_scales': True}, 'phase': 'forward'}, {}, {'provider': 'webgpu', 'target': 'webgpu-portable-v1'}, {'axes': ['key_rows', 'granule'], 'contract': 'reference', 'defaults': {'granule': 128, 'key_rows': 32}, 'factory': 'packed_channel_partial', 'identifier': 'attention.webgpu.packed_decode', 'kind': 'attention_adaptive_partial', 'module': 'akas.ops.attention.webgpu', 'phase': 'forward', 'provider': 'webgpu', 'semantic_keys': ['r', 'h', 'kh', 'd', 'capacity', 'splits', 'kv_dtype'], 'targets': []})}
