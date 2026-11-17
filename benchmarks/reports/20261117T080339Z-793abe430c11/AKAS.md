# akas speech measurements

[Overview](README.md) · [Comparison](COMPARISON.md) · [AKAS](AKAS.md) · [Baselines](BASELINES.md) · [Measurements](measurements.csv)

Timings cover one warm localhost request through receipt of complete PCM16. Wall time and real-time factor (RTF, wall time divided by audio duration) are per-utterance medians. Comparison ratios use geometric means of matched successful utterances. Pass counts cover request completion, format and non-silent audio. AKAS also reports end of speech. WAV byte determinism is assessed within each runtime. Both runtimes used the same case-specific presets. Intelligibility, pronunciation and MOS were outside the measurement scope.

Model Zyphra/ZONOS3-Mini · q8_0 · 44,100 Hz · case-specific voices are listed in the overview.
Run [20261117080339952097-5ce67fc9](https://github.com/kreasof-ai/akas/blob/4c7bfaa20d59b1a4e0ff0c81c8de14e73dbcf1d7/benchmarks/runs/20261117080339952097-5ce67fc9/run.json) · report SHA-256 `c8fb26b829a304e589acd0758abfff9e0bb30504c04b70f4978f1170a535bfc7`.
Checkpoint SHA-256 `60568efc88e2f9231f4f5c061e246421f575986642d72ab9ae68f77d6cf76be4`. Projector SHA-256 `aeeb92325b4b6927bef656390b78e1246148065f947d8b59c2663557a809414d`.
Compiled program SHA-256 `d699af0398d274bfa72738600bcbaa50ad0859a627cbc4915bfac24684408184`.
Compiled codec frame capacities: 16, 32, 64, 128, 256, 384.

| Language | Case | Tier | Status | Repeats | Wall ms | Audio s | RTF | PCM repeatable | Example | Error |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| en | en-0007 | short | ok | 2 | 871.6 | 2.733 | 0.319 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/4c7bfaa20d59b1a4e0ff0c81c8de14e73dbcf1d7/benchmarks/runs/20261117080339952097-5ce67fc9/audio/01-01.wav) |  |
| en | en-0320 | short | ok | 2 | 761.2 | 2.600 | 0.293 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/4c7bfaa20d59b1a4e0ff0c81c8de14e73dbcf1d7/benchmarks/runs/20261117080339952097-5ce67fc9/audio/02-01.wav) |  |
| en | en-0942 | medium | ok | 2 | 922.5 | 3.400 | 0.271 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/4c7bfaa20d59b1a4e0ff0c81c8de14e73dbcf1d7/benchmarks/runs/20261117080339952097-5ce67fc9/audio/03-01.wav) |  |
| en | en-0030 | medium | ok | 2 | 1055.7 | 4.400 | 0.240 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/4c7bfaa20d59b1a4e0ff0c81c8de14e73dbcf1d7/benchmarks/runs/20261117080339952097-5ce67fc9/audio/04-01.wav) |  |
| en | en-0082 | long | ok | 2 | 1248.0 | 5.800 | 0.215 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/4c7bfaa20d59b1a4e0ff0c81c8de14e73dbcf1d7/benchmarks/runs/20261117080339952097-5ce67fc9/audio/05-01.wav) |  |
| en | en-0977 | long | ok | 2 | 1345.2 | 6.600 | 0.204 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/4c7bfaa20d59b1a4e0ff0c81c8de14e73dbcf1d7/benchmarks/runs/20261117080339952097-5ce67fc9/audio/06-01.wav) |  |
| fr | fr-0007 | short | ok | 2 | 1287.8 | 5.200 | 0.248 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/4c7bfaa20d59b1a4e0ff0c81c8de14e73dbcf1d7/benchmarks/runs/20261117080339952097-5ce67fc9/audio/07-01.wav) |  |
| fr | fr-0320 | short | ok | 2 | 1405.2 | 5.533 | 0.254 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/4c7bfaa20d59b1a4e0ff0c81c8de14e73dbcf1d7/benchmarks/runs/20261117080339952097-5ce67fc9/audio/08-01.wav) |  |
| fr | fr-0942 | medium | ok | 2 | 1721.9 | 7.200 | 0.239 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/4c7bfaa20d59b1a4e0ff0c81c8de14e73dbcf1d7/benchmarks/runs/20261117080339952097-5ce67fc9/audio/09-01.wav) |  |
| fr | fr-0030 | medium | ok | 2 | 2079.6 | 8.600 | 0.242 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/4c7bfaa20d59b1a4e0ff0c81c8de14e73dbcf1d7/benchmarks/runs/20261117080339952097-5ce67fc9/audio/10-01.wav) |  |
| fr | fr-0082 | long | ok | 2 | 2970.2 | 12.400 | 0.240 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/4c7bfaa20d59b1a4e0ff0c81c8de14e73dbcf1d7/benchmarks/runs/20261117080339952097-5ce67fc9/audio/11-01.wav) |  |
| fr | fr-0977 | long | ok | 2 | 2441.8 | 10.933 | 0.223 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/4c7bfaa20d59b1a4e0ff0c81c8de14e73dbcf1d7/benchmarks/runs/20261117080339952097-5ce67fc9/audio/12-01.wav) |  |
| es | es-0007 | short | ok | 2 | 667.4 | 2.200 | 0.303 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/4c7bfaa20d59b1a4e0ff0c81c8de14e73dbcf1d7/benchmarks/runs/20261117080339952097-5ce67fc9/audio/13-01.wav) |  |
| es | es-0320 | short | ok | 2 | 931.9 | 3.667 | 0.254 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/4c7bfaa20d59b1a4e0ff0c81c8de14e73dbcf1d7/benchmarks/runs/20261117080339952097-5ce67fc9/audio/14-01.wav) |  |
| es | es-0942 | medium | ok | 2 | 1212.9 | 4.333 | 0.280 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/4c7bfaa20d59b1a4e0ff0c81c8de14e73dbcf1d7/benchmarks/runs/20261117080339952097-5ce67fc9/audio/15-01.wav) |  |
| es | es-0030 | medium | ok | 2 | 1244.9 | 5.400 | 0.231 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/4c7bfaa20d59b1a4e0ff0c81c8de14e73dbcf1d7/benchmarks/runs/20261117080339952097-5ce67fc9/audio/16-01.wav) |  |
| es | es-0082 | long | ok | 2 | 1435.0 | 6.733 | 0.213 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/4c7bfaa20d59b1a4e0ff0c81c8de14e73dbcf1d7/benchmarks/runs/20261117080339952097-5ce67fc9/audio/17-01.wav) |  |
| es | es-0977 | long | ok | 2 | 1826.5 | 9.133 | 0.200 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/4c7bfaa20d59b1a4e0ff0c81c8de14e73dbcf1d7/benchmarks/runs/20261117080339952097-5ce67fc9/audio/18-01.wav) |  |
| de | de-0007 | short | ok | 2 | 1003.9 | 3.867 | 0.260 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/4c7bfaa20d59b1a4e0ff0c81c8de14e73dbcf1d7/benchmarks/runs/20261117080339952097-5ce67fc9/audio/19-01.wav) |  |
| de | de-0320 | short | ok | 2 | 911.3 | 3.533 | 0.258 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/4c7bfaa20d59b1a4e0ff0c81c8de14e73dbcf1d7/benchmarks/runs/20261117080339952097-5ce67fc9/audio/20-01.wav) |  |
| de | de-0942 | medium | ok | 2 | 1166.6 | 4.133 | 0.282 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/4c7bfaa20d59b1a4e0ff0c81c8de14e73dbcf1d7/benchmarks/runs/20261117080339952097-5ce67fc9/audio/21-01.wav) |  |
| de | de-0030 | medium | ok | 2 | 1113.7 | 4.733 | 0.235 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/4c7bfaa20d59b1a4e0ff0c81c8de14e73dbcf1d7/benchmarks/runs/20261117080339952097-5ce67fc9/audio/22-01.wav) |  |
| de | de-0082 | long | ok | 2 | 1938.5 | 8.467 | 0.229 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/4c7bfaa20d59b1a4e0ff0c81c8de14e73dbcf1d7/benchmarks/runs/20261117080339952097-5ce67fc9/audio/23-01.wav) |  |
| de | de-0977 | long | ok | 2 | 1591.3 | 7.467 | 0.213 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/4c7bfaa20d59b1a4e0ff0c81c8de14e73dbcf1d7/benchmarks/runs/20261117080339952097-5ce67fc9/audio/24-01.wav) |  |
| it | it-0007 | short | ok | 2 | 905.6 | 3.200 | 0.283 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/4c7bfaa20d59b1a4e0ff0c81c8de14e73dbcf1d7/benchmarks/runs/20261117080339952097-5ce67fc9/audio/25-01.wav) |  |
| it | it-0320 | short | ok | 2 | 1107.4 | 4.267 | 0.260 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/4c7bfaa20d59b1a4e0ff0c81c8de14e73dbcf1d7/benchmarks/runs/20261117080339952097-5ce67fc9/audio/26-01.wav) |  |
| it | it-0942 | medium | ok | 2 | 1262.5 | 4.400 | 0.287 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/4c7bfaa20d59b1a4e0ff0c81c8de14e73dbcf1d7/benchmarks/runs/20261117080339952097-5ce67fc9/audio/27-01.wav) |  |
| it | it-0030 | medium | ok | 2 | 1485.9 | 6.867 | 0.216 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/4c7bfaa20d59b1a4e0ff0c81c8de14e73dbcf1d7/benchmarks/runs/20261117080339952097-5ce67fc9/audio/28-01.wav) |  |
| it | it-0082 | long | ok | 2 | 1910.1 | 8.000 | 0.239 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/4c7bfaa20d59b1a4e0ff0c81c8de14e73dbcf1d7/benchmarks/runs/20261117080339952097-5ce67fc9/audio/29-01.wav) |  |
| it | it-0977 | long | ok | 2 | 1877.5 | 8.600 | 0.218 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/4c7bfaa20d59b1a4e0ff0c81c8de14e73dbcf1d7/benchmarks/runs/20261117080339952097-5ce67fc9/audio/30-01.wav) |  |
| pt | pt-0007 | short | ok | 2 | 876.6 | 3.467 | 0.253 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/4c7bfaa20d59b1a4e0ff0c81c8de14e73dbcf1d7/benchmarks/runs/20261117080339952097-5ce67fc9/audio/31-01.wav) |  |
| pt | pt-0320 | short | ok | 2 | 992.0 | 3.800 | 0.261 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/4c7bfaa20d59b1a4e0ff0c81c8de14e73dbcf1d7/benchmarks/runs/20261117080339952097-5ce67fc9/audio/32-01.wav) |  |
| pt | pt-0942 | medium | ok | 2 | 1094.3 | 4.200 | 0.261 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/4c7bfaa20d59b1a4e0ff0c81c8de14e73dbcf1d7/benchmarks/runs/20261117080339952097-5ce67fc9/audio/33-01.wav) |  |
| pt | pt-0030 | medium | ok | 2 | 1211.5 | 5.067 | 0.239 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/4c7bfaa20d59b1a4e0ff0c81c8de14e73dbcf1d7/benchmarks/runs/20261117080339952097-5ce67fc9/audio/34-01.wav) |  |
| pt | pt-0082 | long | ok | 2 | 1797.9 | 8.600 | 0.209 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/4c7bfaa20d59b1a4e0ff0c81c8de14e73dbcf1d7/benchmarks/runs/20261117080339952097-5ce67fc9/audio/35-01.wav) |  |
| pt | pt-0977 | long | ok | 2 | 1602.8 | 8.133 | 0.197 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/4c7bfaa20d59b1a4e0ff0c81c8de14e73dbcf1d7/benchmarks/runs/20261117080339952097-5ce67fc9/audio/36-01.wav) |  |
| zh | zh-0007 | short | ok | 2 | 832.0 | 2.933 | 0.284 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/4c7bfaa20d59b1a4e0ff0c81c8de14e73dbcf1d7/benchmarks/runs/20261117080339952097-5ce67fc9/audio/37-01.wav) |  |
| zh | zh-0320 | short | ok | 2 | 874.0 | 3.467 | 0.252 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/4c7bfaa20d59b1a4e0ff0c81c8de14e73dbcf1d7/benchmarks/runs/20261117080339952097-5ce67fc9/audio/38-01.wav) |  |
| zh | zh-0942 | medium | ok | 2 | 884.9 | 3.200 | 0.277 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/4c7bfaa20d59b1a4e0ff0c81c8de14e73dbcf1d7/benchmarks/runs/20261117080339952097-5ce67fc9/audio/39-01.wav) |  |
| zh | zh-0030 | medium | ok | 2 | 886.2 | 3.333 | 0.266 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/4c7bfaa20d59b1a4e0ff0c81c8de14e73dbcf1d7/benchmarks/runs/20261117080339952097-5ce67fc9/audio/40-01.wav) |  |
| zh | zh-0082 | long | ok | 2 | 1304.0 | 4.933 | 0.264 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/4c7bfaa20d59b1a4e0ff0c81c8de14e73dbcf1d7/benchmarks/runs/20261117080339952097-5ce67fc9/audio/41-01.wav) |  |
| zh | zh-0977 | long | ok | 2 | 1188.5 | 5.200 | 0.229 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/4c7bfaa20d59b1a4e0ff0c81c8de14e73dbcf1d7/benchmarks/runs/20261117080339952097-5ce67fc9/audio/42-01.wav) |  |
| ko | ko-0007 | short | ok | 2 | 986.7 | 3.800 | 0.260 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/4c7bfaa20d59b1a4e0ff0c81c8de14e73dbcf1d7/benchmarks/runs/20261117080339952097-5ce67fc9/audio/43-01.wav) |  |
| ko | ko-0320 | short | ok | 2 | 1034.2 | 4.267 | 0.242 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/4c7bfaa20d59b1a4e0ff0c81c8de14e73dbcf1d7/benchmarks/runs/20261117080339952097-5ce67fc9/audio/44-01.wav) |  |
| ko | ko-0942 | medium | ok | 2 | 951.0 | 3.467 | 0.274 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/4c7bfaa20d59b1a4e0ff0c81c8de14e73dbcf1d7/benchmarks/runs/20261117080339952097-5ce67fc9/audio/45-01.wav) |  |
| ko | ko-0030 | medium | ok | 2 | 793.8 | 2.467 | 0.322 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/4c7bfaa20d59b1a4e0ff0c81c8de14e73dbcf1d7/benchmarks/runs/20261117080339952097-5ce67fc9/audio/46-01.wav) |  |
| ko | ko-0082 | long | ok | 2 | 1337.1 | 6.067 | 0.220 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/4c7bfaa20d59b1a4e0ff0c81c8de14e73dbcf1d7/benchmarks/runs/20261117080339952097-5ce67fc9/audio/47-01.wav) |  |
| ko | ko-0977 | long | ok | 2 | 1825.8 | 8.467 | 0.216 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/4c7bfaa20d59b1a4e0ff0c81c8de14e73dbcf1d7/benchmarks/runs/20261117080339952097-5ce67fc9/audio/48-01.wav) |  |
| hi | hi-0007 | short | ok | 2 | 1275.4 | 5.067 | 0.252 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/4c7bfaa20d59b1a4e0ff0c81c8de14e73dbcf1d7/benchmarks/runs/20261117080339952097-5ce67fc9/audio/49-01.wav) |  |
| hi | hi-0320 | short | ok | 2 | 937.4 | 3.667 | 0.256 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/4c7bfaa20d59b1a4e0ff0c81c8de14e73dbcf1d7/benchmarks/runs/20261117080339952097-5ce67fc9/audio/50-01.wav) |  |
| hi | hi-0942 | medium | ok | 2 | 981.2 | 4.067 | 0.241 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/4c7bfaa20d59b1a4e0ff0c81c8de14e73dbcf1d7/benchmarks/runs/20261117080339952097-5ce67fc9/audio/51-01.wav) |  |
| hi | hi-0030 | medium | ok | 2 | 1273.1 | 5.200 | 0.245 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/4c7bfaa20d59b1a4e0ff0c81c8de14e73dbcf1d7/benchmarks/runs/20261117080339952097-5ce67fc9/audio/52-01.wav) |  |
| hi | hi-0082 | long | ok | 2 | 1592.7 | 7.867 | 0.202 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/4c7bfaa20d59b1a4e0ff0c81c8de14e73dbcf1d7/benchmarks/runs/20261117080339952097-5ce67fc9/audio/53-01.wav) |  |
| hi | hi-0977 | long | ok | 2 | 1642.0 | 7.467 | 0.220 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/4c7bfaa20d59b1a4e0ff0c81c8de14e73dbcf1d7/benchmarks/runs/20261117080339952097-5ce67fc9/audio/54-01.wav) |  |
| te | te-0007 | short | ok | 2 | 1117.7 | 3.733 | 0.299 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/4c7bfaa20d59b1a4e0ff0c81c8de14e73dbcf1d7/benchmarks/runs/20261117080339952097-5ce67fc9/audio/55-01.wav) |  |
| te | te-0320 | short | ok | 2 | 1267.1 | 4.400 | 0.288 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/4c7bfaa20d59b1a4e0ff0c81c8de14e73dbcf1d7/benchmarks/runs/20261117080339952097-5ce67fc9/audio/56-01.wav) |  |
| te | te-0942 | medium | ok | 2 | 958.6 | 3.400 | 0.282 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/4c7bfaa20d59b1a4e0ff0c81c8de14e73dbcf1d7/benchmarks/runs/20261117080339952097-5ce67fc9/audio/57-01.wav) |  |
| te | te-0030 | medium | ok | 2 | 1428.1 | 6.533 | 0.219 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/4c7bfaa20d59b1a4e0ff0c81c8de14e73dbcf1d7/benchmarks/runs/20261117080339952097-5ce67fc9/audio/58-01.wav) |  |
| te | te-0082 | long | ok | 2 | 1372.4 | 6.067 | 0.226 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/4c7bfaa20d59b1a4e0ff0c81c8de14e73dbcf1d7/benchmarks/runs/20261117080339952097-5ce67fc9/audio/59-01.wav) |  |
| te | te-0977 | long | ok | 2 | 2623.8 | 12.067 | 0.217 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/4c7bfaa20d59b1a4e0ff0c81c8de14e73dbcf1d7/benchmarks/runs/20261117080339952097-5ce67fc9/audio/60-01.wav) |  |
