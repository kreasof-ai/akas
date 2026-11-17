# Speech benchmark comparison

[Overview](README.md) · [Comparison](COMPARISON.md) · [AKAS](AKAS.md) · [Baselines](BASELINES.md) · [Measurements](measurements.csv)

Timings cover one warm localhost request through receipt of complete PCM16. Wall time and real-time factor (RTF, wall time divided by audio duration) are per-utterance medians. Comparison ratios use geometric means of matched successful utterances. Pass counts cover request completion, format and non-silent audio. AKAS also reports end of speech. WAV byte determinism is assessed within each runtime. Both runtimes used the same case-specific presets. Intelligibility, pronunciation and MOS were outside the measurement scope.

| Language | Cases | AKAS passed | Baseline passed | Paired | Wall wins | Wall speedup | Audio throughput speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| English (`en`) | 6 | 6 | 6 | 6 | 6 | 1.695× | 1.752× |
| French (`fr`) | 6 | 6 | 6 | 6 | 6 | 1.437× | 1.513× |
| Spanish (`es`) | 6 | 6 | 6 | 6 | 6 | 1.580× | 1.562× |
| German (`de`) | 6 | 6 | 6 | 6 | 6 | 1.627× | 1.662× |
| Italian (`it`) | 6 | 6 | 6 | 6 | 6 | 1.566× | 1.582× |
| Portuguese (`pt`) | 6 | 6 | 6 | 6 | 6 | 1.650× | 1.684× |
| Mandarin Chinese (`zh`) | 6 | 6 | 6 | 6 | 6 | 1.537× | 1.639× |
| Korean (`ko`) | 6 | 6 | 6 | 6 | 6 | 1.449× | 1.403× |
| Hindi (`hi`) | 6 | 6 | 6 | 6 | 6 | 1.398× | 1.430× |
| Telugu (`te`) | 6 | 6 | 6 | 6 | 6 | 1.567× | 1.561× |
| **All languages** | 60 | 60 | 60 | 60 | 60 | 1.548× | 1.575× |
