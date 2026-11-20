"""WebGPU bindings for shared spectral computations."""

from .kernel import (
    band_gather,
    band_scatter,
    channel_affine,
    channel_slice,
    complex_correction,
    exclusive_attention,
    elementwise_binary,
    istft_matrix,
    istft_overlap,
    spectrum_band,
    stft,
    token_scatter,
    token_transpose,
)
from .definition import IMPLEMENTATIONS as _CHOICES

IMPLEMENTATIONS = tuple(choice for choice in _CHOICES if choice.provider == "webgpu")
