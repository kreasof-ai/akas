# DSL implementation SHA256: e3ed373b5cc16bc3678b9cd2752c0d38a4cf63e5930a2c1568a1f99bf0389bd5
from akas.kernels.definition import define

def tensor_export():
    return {"kernel": define({'contract': 'reference', 'kind': 'rms', 'parameters': {'c': 2048, 'eps': 1e-05, 'r': 1}, 'phase': 'decode'}, {}, {'provider': 'cuda', 'target': 'sm_90a'}, {'axes': [], 'contract': 'reference', 'defaults': {'algorithm': 'serial-tree'}, 'factory': 'rms', 'identifier': 'normalization.kernel', 'kind': 'rms', 'module': 'akas.ops.normalization.kernel', 'phase': 'forward', 'provider': 'cuda', 'semantic_keys': None, 'targets': []})}
