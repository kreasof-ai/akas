from tensor.compiler.cuda.matmul import matmul
def tensor_export():
 return dict(kernel=matmul(32, 65536, 2048, dtype='float16', out_dtype='float32', block_m=32, block_n=128, block_k=64, threads=128, stages=2))
