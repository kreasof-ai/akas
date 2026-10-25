import sys
sys.path.insert(0,"/home/sagemaker-user/akas/build/lfm-moe")
from q4_paired_prefill import factory
from akas.kernels.definition import KernelSpec
def tensor_export():
 return {"kernel":factory(KernelSpec("quantized_experts",{'r': 4096, 'top': 4, 'experts': 64, 'k': 2048, 'o': 1536, 'input_dtype': 'float32', 'routed_input': False, 'round_inputs': True, 'tile_map': True, 'weight_parts': ({'rows': 98304, 'format': 'ggml', 'options': {'encoding': 2}},), 'block_m': 128, 'columns': 64, 'block_k': 128, 'threads': 256}),{})}
