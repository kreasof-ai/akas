# DSL implementation SHA256: f5991dce0191643680b284fc6fad3185af26b6611be795dac16f2d64c3a61587
from akas.kernels.definition import define

def tensor_export():
    return {"kernel": define({'contract': 'reference', 'kind': 'routed_combine', 'parameters': {'c': 2048, 'input_dtype': 'bfloat16', 'r': 1, 'top': 4}, 'phase': 'decode'}, {}, {'provider': 'cuda', 'target': 'sm_90a'}, {'axes': [], 'contract': 'reference', 'defaults': {}, 'factory': 'routed_combine', 'identifier': 'akas.ops.moe.kernel:routed_combine', 'kind': 'routed_combine', 'module': 'akas.ops.moe.kernel', 'phase': 'forward', 'provider': 'cuda', 'semantic_keys': None, 'targets': []})}
