from akas.convolution_tile_compat.factory import make
def tensor_export():
 return {"kernel":make({'r': 3, 'c': 513, 'chunk': 65, 'length': 5, 'columns': 32, 'threads': 128, 'time_tile': 32},False)}
