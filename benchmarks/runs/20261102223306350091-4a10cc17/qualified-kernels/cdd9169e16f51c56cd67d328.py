# DSL implementation SHA256: 56e4cebd04efdba56e0ea93c384d1573636a9a758bcba4fe488fa27911bbd40a
from akas.compilation.transforms.state import make_kernel

def tensor_export():
    return {"kernel": make_kernel('akas.kernels.definition', 'define', [{'contract': 'reference', 'kind': 'cache_write', 'parameters': {'capacity': 2176, 'd': 64, 'h': 32, 'kh': 8, 'kv_dtype': 'float16', 'optimized': True, 'provider': 'cuda', 'r': 1, 'unit_scales': True}, 'phase': 'decode'}, {}, {'provider': 'cuda', 'target': 'sm_90a'}, {'axes': [], 'contract': 'reference', 'defaults': {}, 'factory': 'cache_write', 'identifier': 'attention.reference', 'kind': 'cache_write', 'module': 'akas.ops.attention.kernel', 'phase': 'forward', 'provider': 'cuda', 'semantic_keys': None, 'targets': []}], [2, 3, 4, 5, 6], 1, 256)}
