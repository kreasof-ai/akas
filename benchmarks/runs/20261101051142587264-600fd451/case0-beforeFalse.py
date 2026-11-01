from akas.typed_activation_compat.factory import make
def tensor_export():
 return {"kernel":make('add',{'r': 33, 'c': 513},False)}
