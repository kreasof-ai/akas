# DSL implementation SHA256: 3e6ce17f517108deca75e6bef3930beb3500d59c9b423a0706519b311d93e8fb
from akas.kernels.definition import define

def tensor_export():
    return {"kernel": define({'contract': 'reference', 'kind': 'swiglu', 'parameters': {'c': 11776, 'r': 1}, 'phase': 'decode'}, {}, {'provider': 'cuda', 'target': 'sm_90a'}, {'axes': [], 'contract': 'reference', 'defaults': {'flat': True, 'input_names': ['x', 'y']}, 'factory': 'swiglu', 'identifier': 'activation.kernel', 'kind': 'swiglu', 'module': 'akas.ops.activation.kernel', 'phase': 'forward', 'provider': 'cuda', 'semantic_keys': None, 'targets': []})}
