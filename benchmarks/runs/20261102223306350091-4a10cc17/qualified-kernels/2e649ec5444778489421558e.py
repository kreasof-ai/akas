# DSL implementation SHA256: f5991dce0191643680b284fc6fad3185af26b6611be795dac16f2d64c3a61587
from akas.kernels.definition import define

def tensor_export():
    return {"kernel": define({'contract': 'reference', 'kind': 'quantized_routes', 'parameters': {'experts': 64, 'r': 1, 'top': 4}, 'phase': 'decode'}, {}, {'provider': 'cuda', 'target': 'sm_90a'}, {'axes': [], 'contract': 'reference', 'defaults': {}, 'factory': 'quantized_routes', 'identifier': 'akas.ops.moe.kernel:quantized_routes', 'kind': 'quantized_routes', 'module': 'akas.ops.moe.kernel', 'phase': 'forward', 'provider': 'cuda', 'semantic_keys': None, 'targets': []})}
