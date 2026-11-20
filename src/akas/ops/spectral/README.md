# Spectral and band-layout operations

These fixed-shape operations cover centered real STFT/iSTFT, bounded complex
mask-plus-residual correction, exclusive self-attention, and explicit
band/frame data layout changes. All arithmetic and FFT accumulation is F32.

The WebGPU implementations are portable scalar kernels. Twiddles use the full
`exp(±2πi × frequency × tap / n_fft)` phase, not quarter-turn-only values.
STFT evaluates only `n_fft // 2 + 1` nonnegative frequencies. The inverse uses
an F32 host-derived real-DFT matrix followed by conjugate-symmetric frame
reconstruction, windowed overlap-add and window-squared normalization.
Short inputs
use zero padding while longer inputs use the same reflection policy as
`torch.stft`. Graph geometry fixes exact sample and frame counts so a short
clip never receives fictitious future audio from an eight-second zero pad.

CUDA has no registered implementation yet. Host validation uses NumPy FFT and
independent tensor formulas; it does not invoke a Torch model or a device
kernel as its reference.
