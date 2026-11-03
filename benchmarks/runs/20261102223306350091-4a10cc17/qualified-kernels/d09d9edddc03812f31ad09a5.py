# DSL implementation SHA256: 56e4cebd04efdba56e0ea93c384d1573636a9a758bcba4fe488fa27911bbd40a
from akas.compilation.transforms.state import make_kernel

def tensor_export():
    return {"kernel": make_kernel('akas.kernels.definition', 'define', [{'contract': 'reference', 'kind': 'rope', 'parameters': {'d': 64, 'h': 32, 'position_limit': 2176, 'provider': 'cuda', 'r': 1, 'theta': 10000000.0}, 'phase': 'decode'}, {}, {'provider': 'cuda', 'target': 'sm_90a'}, {'axes': [], 'contract': 'reference', 'defaults': {}, 'factory': 'rope', 'identifier': 'attention.reference', 'kind': 'rope', 'module': 'akas.ops.attention.kernel', 'phase': 'forward', 'provider': 'cuda', 'semantic_keys': None, 'targets': []}], [2], 1, 256)}
