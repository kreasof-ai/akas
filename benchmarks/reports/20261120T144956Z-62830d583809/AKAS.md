# AKAS enhancement measurements

[Overview](README.md) · [Comparison](COMPARISON.md) · [AKAS](AKAS.md) · [Baselines](BASELINES.md) · [Measurements](measurements.csv)

Warm wall time covers WAV decoding and complete offline enhancement. Model loading and WAV writing are excluded. RTF is wall seconds per input audio second. These public Inworld demo/golden clips are not the private VIBE evaluation set. Golden error is against the independent training-reference output; other errors are against the upstream CPU runtime. Operation and golden checks are not an independent VIBE quality evaluation. The upstream CPU comparison covers only cases completed and validated by both runtimes.

Run [20261120144956732293-6995a945](https://github.com/kreasof-ai/akas/blob/a3947ffa4d7ad10fa59f0c791cef60d4b401d090/benchmarks/runs/20261120144956732293-6995a945/run.json). Report SHA-256 `7851830328b744614d267de04ac1adf3be70d434c100a909dbb8a82197bf3ed8`.
Model Inworld Voice Isolator v1 (offline). Source checkpoint `f32`. Execution storage `float32`.
Checkpoint SHA-256 `647335a6aeb4dc706b895532c209d51e0e84a4849d17b339f00d4392a050ae53`. Each clip has 1 warmup and 3 measured requests.
Compiled program SHA-256 `c6597cd30654e08a15a3ddb8907dfd48938acf4154a510f4a9c38a94b7f1ff98`.

| Case | Audio s | Wall s | RTF | Deterministic | Golden max abs | Upstream RMS | Repeats |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| golden-3s | 3.000 | 0.710 | 0.237 | yes | 2.207e-07 | 3.465e-08 | 3 |
| demo-hiss-10s | 10.770 | 5.768 | 0.536 | yes | — | — | 3 |
| demo-noisy-24s | 24.729 | 11.521 | 0.466 | yes | — | — | 3 |

## Retained audio (first measured repeat)

- `golden-3s`: [voice](https://github.com/kreasof-ai/akas/blob/a3947ffa4d7ad10fa59f0c791cef60d4b401d090/benchmarks/runs/20261120144956732293-6995a945/audio/01-02-voice.wav) · [background](https://github.com/kreasof-ai/akas/blob/a3947ffa4d7ad10fa59f0c791cef60d4b401d090/benchmarks/runs/20261120144956732293-6995a945/audio/01-02-background.wav)
- `demo-hiss-10s`: [voice](https://github.com/kreasof-ai/akas/blob/a3947ffa4d7ad10fa59f0c791cef60d4b401d090/benchmarks/runs/20261120144956732293-6995a945/audio/02-02-voice.wav) · [background](https://github.com/kreasof-ai/akas/blob/a3947ffa4d7ad10fa59f0c791cef60d4b401d090/benchmarks/runs/20261120144956732293-6995a945/audio/02-02-background.wav)
- `demo-noisy-24s`: [voice](https://github.com/kreasof-ai/akas/blob/a3947ffa4d7ad10fa59f0c791cef60d4b401d090/benchmarks/runs/20261120144956732293-6995a945/audio/03-02-voice.wav) · [background](https://github.com/kreasof-ai/akas/blob/a3947ffa4d7ad10fa59f0c791cef60d4b401d090/benchmarks/runs/20261120144956732293-6995a945/audio/03-02-background.wav)
