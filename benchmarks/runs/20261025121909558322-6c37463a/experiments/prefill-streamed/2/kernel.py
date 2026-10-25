from akas.ops.linear.webgpu import native_half_pairs
def tensor_export():
    return {"kernel": native_half_pairs({'r': 512, 'k': 2048, 'o': 10752, 'optimized': True, 'ggml': True, 'type': 1, 'half_accum': True, 'input_dtype': 'float16', 'half_tile': (32, 64, 4, 4, 128), 'half_pad': 1, 'half_k': 16, 'half_streamed': False}, paired=True)}
