from akas.typed_activation_compat.factory import make
def tensor_export():
 return {"kernel":make('add_rms',{'r': 4, 'c': 2048, 'eps': 1e-05, 'separate_residual': True},True)}
