# DSL implementation SHA256: 109c6514f0b04ea59031e7cad19f1f9a6d6872f540a99730b42b93e5ba6282de
from akas.compilation.transforms.state import make_kernel

def tensor_export():
    return {"kernel": make_kernel('akas.kernels.definition', 'define', [{'contract': 'reference', 'kind': 'advance', 'parameters': {'r': 1}, 'phase': 'decode'}, {}, {'provider': 'cuda', 'target': 'sm_90a'}, {'axes': [], 'contract': 'reference', 'defaults': {}, 'factory': 'advance', 'identifier': 'sequence.reference', 'kind': 'advance', 'module': 'akas.ops.sequence.kernel', 'phase': 'forward', 'provider': 'cuda', 'semantic_keys': None, 'targets': []}], [0], 1, 256)}
