from akas.convolution_tile_compat.factory import make
def tensor_export():
 return {"kernel":make({'r': 4, 'c': 2048, 'chunk': 2048, 'length': 3, 'columns': 32, 'threads': 256, 'time_tile': 32},False)}
