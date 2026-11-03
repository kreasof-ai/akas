# DSL implementation SHA256: 2f1f2fb14c0a764b5f9ae24879c8eb2c440df565c72ebfea63320f2cf23f2c90
from akas.kernels.definition import define

def tensor_export():
    return {"kernel": define({'contract': 'reference', 'kind': 'rope', 'parameters': {'d': 64, 'h': 8, 'position_limit': 2176, 'provider': 'cuda', 'r': 2048, 'theta': 10000000.0}, 'phase': 'prefill'}, {}, {'provider': 'cuda', 'target': 'sm_90a'}, {'axes': [], 'contract': 'reference', 'defaults': {}, 'factory': 'rope', 'identifier': 'attention.reference', 'kind': 'rope', 'module': 'akas.ops.attention.kernel', 'phase': 'forward', 'provider': 'cuda', 'semantic_keys': None, 'targets': []})}
