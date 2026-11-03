# DSL implementation SHA256: 85b40a7efc466a5322b1f0557f351c323315ac7b018465d233a5ff1bd480a005
from akas.kernels.definition import define

def tensor_export():
    return {"kernel": define({'contract': 'reference', 'kind': 'argmax', 'parameters': {'c': 128000, 'r': 1}, 'phase': 'decode'}, {'threads': 1024}, {'provider': 'cuda', 'target': 'sm_90a'}, {'axes': ['threads'], 'contract': 'reference', 'defaults': {}, 'factory': 'argmax', 'identifier': 'vocabulary.reference', 'kind': 'argmax', 'module': 'akas.ops.vocabulary.kernel', 'phase': 'forward', 'provider': 'cuda', 'semantic_keys': None, 'targets': []})}
