# DSL implementation SHA256: 84484a2d4a43892ab2d94b9b08859e0a3ea4c0f1d9de31ea49b3ea275737352d
from akas.kernels.definition import define

def tensor_export():
    return {"kernel": define({'contract': 'reference', 'kind': 'linear', 'parameters': {'k': 2048, 'o': 64, 'r': 1}, 'phase': 'decode'}, {}, {'provider': 'cuda', 'target': 'sm_90a'}, {'axes': ['threads'], 'contract': 'reference', 'defaults': {'threads': 128}, 'factory': 'floating_warp', 'identifier': 'linear.cuda.floating_warp', 'kind': 'linear', 'module': 'akas.ops.linear.cuda', 'phase': 'forward', 'provider': 'cuda', 'semantic_keys': None, 'targets': []})}
