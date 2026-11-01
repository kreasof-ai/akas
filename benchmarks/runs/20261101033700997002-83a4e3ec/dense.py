# DSL implementation SHA256: 240596c88b970dac2cabab9598283f1d86a369bcd3fe36860b05c00e41e9e645
from akas.kernels.definition import define

def tensor_export():
    return {"kernel": define({'contract': 'reference', 'kind': 'projection_partial', 'parameters': {'block_m': 16, 'columns': 64, 'k': 2048, 'mma_dtype': 'float16', 'o': 2048, 'packed_copy': True, 'partitions': 1, 'r': 2048, 'stages': 1, 'threads': 128}, 'phase': 'forward'}, {}, {'provider': 'cuda', 'target': 'sm_89'}, {'axes': ['columns', 'block_m', 'threads', 'stages', 'packed_copy', 'mma_dtype'], 'contract': 'reference', 'defaults': {}, 'factory': 'fp8_prequantized', 'identifier': 'linear.pipeline:reference', 'kind': 'projection_partial', 'module': 'akas.ops.linear.cuda', 'phase': 'forward', 'provider': 'cuda', 'semantic_keys': ['r', 'k', 'o', 'partitions'], 'targets': []})}
