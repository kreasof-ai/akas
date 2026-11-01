from akas.paired_reduction_compat.factory import make
def tensor_export():
 return {"kernel":make('expert_merge',{'r': 3, 'top': 4, 'o': 256, 'partitions': 4},True,False)}
