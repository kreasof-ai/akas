# Inworld upstream enhancement measurements

[Overview](README.md) · [Comparison](COMPARISON.md) · [AKAS](AKAS.md) · [Baselines](BASELINES.md) · [Measurements](measurements.csv)

Warm wall time covers WAV decoding and complete offline enhancement. Model loading and WAV writing are excluded. RTF is wall seconds per input audio second. These public Inworld demo/golden clips are not the private VIBE evaluation set. Golden error is against the independent training-reference output; other errors are against the upstream CPU runtime. Operation and golden checks are not an independent VIBE quality evaluation. The upstream CPU comparison covers only cases completed and validated by both runtimes.

Run [20261120122845396569-764570f0](https://github.com/kreasof-ai/akas/blob/a3947ffa4d7ad10fa59f0c791cef60d4b401d090/benchmarks/runs/20261120122845396569-764570f0/run.json). Report SHA-256 `acd7574878ca43837aa84de132d7ddc3271aecef218c33b6b90f86c54970ed19`.
Model Inworld Voice Isolator v1 (offline). Source checkpoint `f32`. Execution storage `float32`.
Checkpoint SHA-256 `647335a6aeb4dc706b895532c209d51e0e84a4849d17b339f00d4392a050ae53`. Each clip has 1 warmup and 3 measured requests.
Upstream runtime `1.0.0`; Torch `2.8.0+cpu`.
CPU threads: 6.
Upstream source revision `c73b3fab81a870410cbac50d2f05f25edc4d10aa`.

| Case | Audio s | Wall s | RTF | Deterministic | Golden max abs | Upstream RMS | Repeats |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| golden-3s | 3.000 | 10.154 | 3.385 | yes | 5.960e-08 | — | 3 |

## Retained audio (first measured repeat)

- `golden-3s`: [voice](https://github.com/kreasof-ai/akas/blob/a3947ffa4d7ad10fa59f0c791cef60d4b401d090/benchmarks/runs/20261120122845396569-764570f0/audio/01-02-voice.wav) · [background](https://github.com/kreasof-ai/akas/blob/a3947ffa4d7ad10fa59f0c791cef60d4b401d090/benchmarks/runs/20261120122845396569-764570f0/audio/01-02-background.wav)
