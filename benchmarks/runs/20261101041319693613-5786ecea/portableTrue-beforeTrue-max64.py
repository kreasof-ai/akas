from akas.map_compat.factory import make
def tensor_export():
 return {"kernel":make({'r': 257, 'experts': 7, 'top': 3},True,True),"outputs":["tile_experts","tile_rows"]}
