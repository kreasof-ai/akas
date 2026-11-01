from akas.typed_activation_compat.factory import make
def tensor_export():
 return {"kernel":make('rms',{'r': 33, 'c': 513, 'eps': 1e-05, 'algorithm': 'serial-tree', 'stage_values': False},False)}
