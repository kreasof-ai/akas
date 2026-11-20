"""Contracts for fixed-size spectral and band-layout operations."""

from akas.kernels.contracts import Implementation, OutputABI
from akas.ops.spectral.validation import VALIDATIONS


_OUTPUTS = {
    "stft": ("out",),
    "istft_matrix": ("out",),
    "istft_overlap": ("out",),
    "complex_correction": ("out",),
    "exclusive_attention": ("out",),
    "token_transpose": ("out",),
    "band_gather": ("out",),
    "band_scatter": ("out",),
    "spectrum_band": ("out",),
    "token_scatter": ("features",),
    "channel_slice": ("out",),
    "channel_affine": ("out",),
    "elementwise_binary": ("out",),
}
OUTPUT_ARGUMENTS = {
    "reference": {kind: OutputABI((-1,), names) for kind, names in _OUTPUTS.items()}
}
OUTPUT_ARGUMENTS["reference"]["band_scatter"] = OutputABI((-1,), ("spectrum",))
IMPLEMENTATIONS = tuple(
    Implementation(
        kind,
        "akas.ops.spectral.kernel",
        "reference",
        provider="webgpu",
        factory=kind,
        name=f"spectral.webgpu.{kind}",
        space=(("algorithm", ("direct",)),) if kind == "stft" else (),
        semantic_keys=("samples", "fft", "hop", "frames", "bins", "reflect")
        if kind == "stft"
        else None,
    )
    for kind in _OUTPUTS
)
