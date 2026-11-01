from akas.paired_reduction_compat.factory import make
def tensor_export():
 return {"kernel":make('swiglu',{'r': 4, 'c': 513, 'packed': True},False,False)}
