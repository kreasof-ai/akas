# DSL implementation SHA256: f5991dce0191643680b284fc6fad3185af26b6611be795dac16f2d64c3a61587
from akas.kernels.definition import define

def tensor_export():
    return {"kernel": define({'contract': 'reference', 'kind': 'biased_router', 'parameters': {'experts': 64, 'normalize': True, 'r': 1, 'scale': 1.0, 'top': 4}, 'phase': 'decode'}, {}, {'provider': 'cuda', 'target': 'sm_90a'}, {'axes': [], 'contract': 'reference', 'defaults': {}, 'factory': 'biased_router', 'identifier': 'akas.ops.moe.kernel:biased_router', 'kind': 'biased_router', 'module': 'akas.ops.moe.kernel', 'phase': 'forward', 'provider': 'cuda', 'semantic_keys': None, 'targets': []})}
