# DSL implementation SHA256: 85b40a7efc466a5322b1f0557f351c323315ac7b018465d233a5ff1bd480a005
from akas.kernels.definition import define

def tensor_export():
    return {"kernel": define({'contract': 'reference', 'kind': 'embedding', 'parameters': {'c': 2048, 'r': 1, 'start': 0, 'v': 128000, 'weight_dtype': 'float16'}, 'phase': 'decode'}, {}, {'provider': 'cuda', 'target': 'sm_90a'}, {'axes': [], 'contract': 'reference', 'defaults': {}, 'factory': 'embedding', 'identifier': 'vocabulary.reference', 'kind': 'embedding', 'module': 'akas.ops.vocabulary.kernel', 'phase': 'forward', 'provider': 'cuda', 'semantic_keys': None, 'targets': []})}
