# Offline enhancement comparison

[Overview](README.md) · [Comparison](COMPARISON.md) · [AKAS](AKAS.md) · [Baselines](BASELINES.md) · [Measurements](measurements.csv)

Warm wall time covers WAV decoding and complete offline enhancement. Model loading and WAV writing are excluded. RTF is wall seconds per input audio second. These public Inworld demo/golden clips are not the private VIBE evaluation set. Golden error is against the independent training-reference output; other errors are against the upstream CPU runtime. Operation and golden checks are not an independent VIBE quality evaluation. The upstream CPU comparison covers only cases completed and validated by both runtimes.


CPU timing is reported only for the 3-second golden. Upstream ran out of memory on the 10.77-second clip; the 24.73-second CPU clip had no valid measurement. See [upstream issue #5](https://github.com/inworld-ai/voice-isolator/issues/5).

| Case | Audio s | Upstream wall s | AKAS wall s | Wall speedup | Upstream RTF | AKAS RTF | Golden parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| golden-3s | 3.000 | 10.154 | 0.710 | 14.300× | 3.385 | 0.237 | yes |
| demo-hiss-10s | 10.770 | — | 5.768 | — | — | 0.536 | not checked |
| demo-noisy-24s | 24.729 | — | 11.521 | — | — | 0.466 | not checked |

## Peak memory on the 11-second golden

Separate from latency timing: Windows per-process GPU memory counters, with both exact-length graphs loaded. Identical compiled kernels and bit-for-bit identical voice and background WAVs across short → long → short.

| Runtime allocation | Dedicated GPU peak (GiB) | Host working-set peak (MiB) |
| --- | ---: | ---: |
| Separate scratch | 2.931 | 422.3 |
| Shared scratch | 1.309 | 423.2 |

Dedicated GPU memory fell by 55.3%. [Memory run](https://github.com/kreasof-ai/akas/blob/a3947ffa4d7ad10fa59f0c791cef60d4b401d090/benchmarks/runs/20261120144205856363-376bb0a6/report.json).

The shared-scratch build passed the independent 3-second and 11-second golden checks at `1e-4`. These checks do not establish accuracy on VIBE.
