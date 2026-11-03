# DSL implementation SHA256: 62f2d8942d004f7203ae0f1f624d6facb2087d0ae5acd5cca591313bbfc722cb
from akas.compilation.transforms.state import make_kernel

def tensor_export():
    return {"kernel": make_kernel('akas.kernels.definition', 'define', [{'contract': 'reference', 'kind': 'conv', 'parameters': {'c': 2048, 'length': 3, 'r': 1}, 'phase': 'decode'}, {}, {'provider': 'cuda', 'target': 'sm_90a'}, {'axes': [], 'contract': 'reference', 'defaults': {}, 'factory': 'convolution', 'identifier': 'convolution.reference', 'kind': 'conv', 'module': 'akas.ops.convolution.kernel', 'phase': 'forward', 'provider': 'cuda', 'semantic_keys': None, 'targets': []}], [2], 1, 256)}
