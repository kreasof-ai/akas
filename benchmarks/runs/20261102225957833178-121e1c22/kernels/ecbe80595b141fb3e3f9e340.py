# DSL implementation SHA256: 67a5ae0ad18df3c2a5bd1b4aebb32dd7edd7068511dc29a0052692ae2ffb7f40
from akas.compilation.transforms.state import make_kernel

def tensor_export():
    return {"kernel": make_kernel('akas.kernels.definition', 'define', [{'contract': 'reference', 'kind': 'rope', 'parameters': {'d': 64, 'h': 32, 'position_limit': 2176, 'provider': 'cuda', 'r': 16, 'theta': 10000000.0}, 'phase': 'decode'}, {}, {'provider': 'cuda', 'target': 'sm_90a'}, {'axes': [], 'contract': 'reference', 'defaults': {}, 'factory': 'rope', 'identifier': 'attention.reference', 'kind': 'rope', 'module': 'akas.ops.attention.kernel', 'phase': 'forward', 'provider': 'cuda', 'semantic_keys': None, 'targets': []}], [2], 16, 256)}
