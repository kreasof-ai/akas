import sys
sys.path.insert(0,"/home/sagemaker-user/akas/build/lfm-moe")
from q4_async_word_prefill import factory
from akas.kernels.definition import KernelSpec
def tensor_export():
 return {"kernel":factory(KernelSpec("quantized_experts",{'r': 4096, 'top': 4, 'experts': 64, 'k': 1536, 'o': 2048, 'input_dtype': 'float32', 'routed_input': True, 'round_inputs': True, 'tile_map': True, 'weight_parts': ({'rows': 131072, 'format': 'ggml', 'options': {'encoding': 2}},), 'block_m': 64, 'columns': 128, 'block_k': 64, 'threads': 128, 'stages': 3}),{})}
