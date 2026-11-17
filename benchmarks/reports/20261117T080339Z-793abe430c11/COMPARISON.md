# Speech benchmark comparison

[Overview](README.md) · [Comparison](COMPARISON.md) · [AKAS](AKAS.md) · [Baselines](BASELINES.md) · [Measurements](measurements.csv)

Timings cover one warm localhost request through receipt of complete PCM16. Wall time and real-time factor (RTF, wall time divided by audio duration) are per-utterance medians. Comparison ratios use geometric means of matched successful utterances. Pass counts cover request completion, format and non-silent audio. AKAS also reports end of speech. WAV byte determinism is assessed within each runtime. Both runtimes used the same case-specific presets. Intelligibility, pronunciation and MOS were outside the measurement scope.

| Language | Cases | AKAS passed | Baseline passed | Paired | Wall wins | Wall speedup | Audio throughput speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| English (`en`) | 6 | 6 | 6 | 6 | 6 | 1.466× | 1.522× |
| French (`fr`) | 6 | 6 | 6 | 6 | 6 | 1.360× | 1.392× |
| Spanish (`es`) | 6 | 6 | 6 | 6 | 6 | 1.475× | 1.426× |
| German (`de`) | 6 | 6 | 6 | 6 | 6 | 1.487× | 1.471× |
| Italian (`it`) | 6 | 6 | 6 | 6 | 6 | 1.388× | 1.390× |
| Portuguese (`pt`) | 6 | 6 | 6 | 6 | 6 | 1.486× | 1.493× |
| Mandarin Chinese (`zh`) | 6 | 6 | 6 | 6 | 6 | 1.379× | 1.406× |
| Korean (`ko`) | 6 | 6 | 6 | 6 | 5 | 1.367× | 1.495× |
| Hindi (`hi`) | 6 | 6 | 6 | 6 | 6 | 1.518× | 1.487× |
| Telugu (`te`) | 6 | 6 | 6 | 6 | 6 | 1.359× | 1.358× |
| **All languages** | 60 | 60 | 60 | 60 | 59 | 1.427× | 1.443× |
