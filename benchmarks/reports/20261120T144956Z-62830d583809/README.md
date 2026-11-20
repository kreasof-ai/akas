# Inworld Voice Isolator: offline enhancement

[Overview](README.md) · [Comparison](COMPARISON.md) · [AKAS](AKAS.md) · [Baselines](BASELINES.md) · [Measurements](measurements.csv)

Workload `voice-enhancement-demo-v1` uses 3 fixed clips. Protocol `warm-voice-enhancement-v1`.

Warm wall time covers WAV decoding and complete offline enhancement. Model loading and WAV writing are excluded. RTF is wall seconds per input audio second. These public Inworld demo/golden clips are not the private VIBE evaluation set. Golden error is against the independent training-reference output; other errors are against the upstream CPU runtime. Operation and golden checks are not an independent VIBE quality evaluation. The upstream CPU comparison covers only cases completed and validated by both runtimes.

| Runtime | Device | Complete and deterministic cases | Golden parity | Run |
| --- | --- | ---: | ---: | --- |
| AKAS (webgpu) | AMD Radeon RX 6700 XT | 3/3 | 1/1 | [20261120144956732293-6995a945](https://github.com/kreasof-ai/akas/blob/a3947ffa4d7ad10fa59f0c791cef60d4b401d090/benchmarks/runs/20261120144956732293-6995a945/run.json) |
| Inworld upstream (cpu) | AMD64 Family 25 Model 33 Stepping 2, AuthenticAMD | 1/1 | 1/1 | [20261120122845396569-764570f0](https://github.com/kreasof-ai/akas/blob/a3947ffa4d7ad10fa59f0c791cef60d4b401d090/benchmarks/runs/20261120122845396569-764570f0/run.json) |

See [Comparison](COMPARISON.md#peak-memory-on-the-11-second-golden) for memory and golden evidence.

## Audio sources and license

The 3-second golden and the hiss demo are derived from LibriSpeech dev-clean utterance `2277-149896-0032` (*Sister Carrie*, read by zinniz; LibriSpeech © 2014 Vassil Panayotov, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)). Inworld added hiss and prepared the golden fixture. The retained outputs are the voice and background separated by each runtime.

[Source recording notice](https://github.com/inworld-ai/voice-isolator/blob/c73b3fab81a870410cbac50d2f05f25edc4d10aa/demo/examples/sources/LICENSE.txt) · [Demo mix recipes](https://github.com/inworld-ai/voice-isolator/blob/c73b3fab81a870410cbac50d2f05f25edc4d10aa/demo/build_examples.py) · [Golden fixture attribution](https://github.com/inworld-ai/voice-isolator/blob/c73b3fab81a870410cbac50d2f05f25edc4d10aa/tests/fixtures/README.md) · [Upstream notices](https://github.com/inworld-ai/voice-isolator/blob/c73b3fab81a870410cbac50d2f05f25edc4d10aa/NOTICE)

- [golden-3s](https://github.com/inworld-ai/voice-isolator/blob/v1.0.0/tests/fixtures/offline_3s/input.wav)
- [demo-hiss-10s](https://github.com/inworld-ai/voice-isolator/blob/v1.0.0/demo/examples/sample_14.wav)
- [demo-noisy-24s](https://github.com/inworld-ai/voice-isolator/blob/v1.0.0/demo/examples/sample_2.wav)
