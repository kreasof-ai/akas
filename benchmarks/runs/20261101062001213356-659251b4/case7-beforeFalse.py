from akas.packed_expert_compat.factory import make
def tensor_export():
 return {"kernel":make('swiglu',{'r': 4, 'c': 513, 'flat': True},False,False)}
