# DSL implementation SHA256: e16e5c77bedaf0d8e2ca2fbfa86eb816c0c615eebe36d81f1594c07dbc881e26
from akas.kernels.definition import define

def tensor_export():
    return {"kernel": define({'contract': 'reference', 'kind': 'next_tokens', 'parameters': {'r': 1}, 'phase': 'decode'}, {}, {'provider': 'cuda', 'target': 'sm_90a'}, {'axes': [], 'contract': 'reference', 'defaults': {}, 'factory': 'next_tokens', 'identifier': 'sequence.reference', 'kind': 'next_tokens', 'module': 'akas.ops.sequence.kernel', 'phase': 'forward', 'provider': 'cuda', 'semantic_keys': None, 'targets': []})}
