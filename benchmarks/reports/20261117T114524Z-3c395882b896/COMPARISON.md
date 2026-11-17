# Speech benchmark comparison

[Overview](README.md) · [Comparison](COMPARISON.md) · [AKAS](AKAS.md) · [Baselines](BASELINES.md) · [Measurements](measurements.csv)

Timings cover one warm localhost request through receipt of complete PCM16. Wall time and real-time factor (RTF, wall time divided by audio duration) are per-utterance medians. Comparison ratios use geometric means of matched successful utterances. Pass counts cover request completion, format and non-silent audio. AKAS also reports end of speech. WAV byte determinism is assessed within each runtime. Both runtimes used the same case-specific presets. Intelligibility, pronunciation and MOS were outside the measurement scope.

| Language | Cases | AKAS passed | Baseline passed | Paired | Wall wins | Wall speedup | Audio throughput speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| English (`en`) | 6 | 6 | 6 | 6 | 6 | 1.673× | 1.721× |
| French (`fr`) | 6 | 6 | 6 | 6 | 6 | 1.580× | 1.524× |
| Spanish (`es`) | 6 | 6 | 6 | 6 | 6 | 1.521× | 1.514× |
| German (`de`) | 6 | 6 | 6 | 6 | 6 | 1.700× | 1.713× |
| Italian (`it`) | 6 | 6 | 6 | 6 | 6 | 1.579× | 1.607× |
| Portuguese (`pt`) | 6 | 6 | 6 | 6 | 6 | 1.501× | 1.552× |
| Mandarin Chinese (`zh`) | 6 | 6 | 6 | 6 | 6 | 1.545× | 1.582× |
| Korean (`ko`) | 6 | 6 | 6 | 6 | 5 | 1.371× | 1.775× |
| Hindi (`hi`) | 6 | 6 | 6 | 6 | 6 | 1.612× | 1.546× |
| Telugu (`te`) | 6 | 6 | 6 | 6 | 6 | 1.397× | 1.428× |
| **All languages** | 60 | 60 | 60 | 60 | 59 | 1.544× | 1.593× |
