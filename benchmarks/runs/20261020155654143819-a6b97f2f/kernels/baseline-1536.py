from akas.ops.moe.cuda import compact_experts
from akas.kernels.definition import KernelSpec
def tensor_export():
 return {"kernel":compact_experts(KernelSpec("quantized_experts",{'r': 4096, 'top': 4, 'experts': 64, 'k': 1536, 'o': 2048, 'input_dtype': 'float32', 'routed_input': True, 'round_inputs': True, 'tile_map': True, 'weight_parts': ({'rows': 131072, 'format': 'ggml', 'options': {'encoding': 2}},), 'block_m': 64, 'columns': 64, 'block_k': 64}),{})}
