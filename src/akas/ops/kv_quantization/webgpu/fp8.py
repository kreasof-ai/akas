"""WebGPU E4M3 uses packed words, requiring no native shader FP8 support."""

from functools import partial
from akas.ops.kv_quantization.portable.fp8 import reference_writer as _writer
from akas.ops.kv_quantization.portable.fp8 import reference_decoder as _decoder

reference_writer = partial(_writer, provider="webgpu")
reference_decoder = partial(_decoder, provider="webgpu")
