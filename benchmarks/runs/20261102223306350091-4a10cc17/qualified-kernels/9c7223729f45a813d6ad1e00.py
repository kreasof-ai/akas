# DSL implementation SHA256: 7bfbe0bb4aa247219db20b9d19d6d7a5b4c4cdbe00aae4937c99431367f069a9
from akas.kernels.definition import define

def tensor_export():
    return {"kernel": define({'contract': 'reference', 'kind': 'expert_tile_map', 'parameters': {'experts': 64, 'max_block_m': 128, 'r': 1, 'top': 4}, 'phase': 'decode'}, {}, {'provider': 'cuda', 'target': 'sm_90a'}, {'axes': [], 'contract': 'reference', 'defaults': {}, 'factory': 'expert_tile_map', 'identifier': 'moe.cuda.parallel_tile_map', 'kind': 'expert_tile_map', 'module': 'akas.ops.moe.cuda', 'phase': 'forward', 'provider': 'cuda', 'semantic_keys': None, 'targets': []})}
