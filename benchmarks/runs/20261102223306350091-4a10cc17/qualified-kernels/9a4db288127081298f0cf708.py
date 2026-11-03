# DSL implementation SHA256: 84484a2d4a43892ab2d94b9b08859e0a3ea4c0f1d9de31ea49b3ea275737352d
from akas.kernels.definition import define

def tensor_export():
    return {"kernel": define({'contract': 'reference', 'kind': 'linear', 'parameters': {'k': 2048, 'o': 128000, 'optimized': True, 'r': 1, 'weight_dtype': 'float16'}, 'phase': 'decode'}, {'algorithm': 'auto', 'block_k': 32, 'block_m': 32, 'block_n': 64, 'maximum_gemv_rows': 31, 'minimum_mma_groups': 32, 'stages': 2, 'threads': 128, 'tma': True}, {'provider': 'cuda', 'target': 'sm_90a'}, {'axes': ['minimum_mma_groups', 'maximum_gemv_rows', 'block_m', 'block_n', 'block_k', 'stages', 'threads', 'algorithm', 'tma'], 'contract': 'reference', 'defaults': {}, 'factory': 'floating_linear', 'identifier': 'linear.cuda.floating', 'kind': 'linear', 'module': 'akas.ops.linear.cuda', 'phase': 'forward', 'provider': 'cuda', 'semantic_keys': None, 'targets': []})}
