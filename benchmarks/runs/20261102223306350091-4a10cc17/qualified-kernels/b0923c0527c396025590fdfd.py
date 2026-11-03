# DSL implementation SHA256: 84484a2d4a43892ab2d94b9b08859e0a3ea4c0f1d9de31ea49b3ea275737352d
from akas.kernels.definition import define

def tensor_export():
    return {"kernel": define({'contract': 'reference', 'kind': 'expert_merge', 'parameters': {'compact': True, 'o': 2048, 'output_dtype': 'bfloat16', 'partitions': 2, 'r': 1, 'routed_input': True, 'top': 4}, 'phase': 'decode'}, {}, {'provider': 'cuda', 'target': 'sm_90a'}, {'axes': [], 'contract': 'reference', 'defaults': {}, 'factory': 'fp8', 'identifier': 'moe.cuda.merge:reference', 'kind': 'expert_merge', 'module': 'akas.ops.linear.cuda', 'phase': 'forward', 'provider': 'cuda', 'semantic_keys': ['r', 'top', 'o', 'partitions', 'output_dtype', 'routed_input'], 'targets': []})}
