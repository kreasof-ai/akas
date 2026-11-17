# akas speech measurements

[Overview](README.md) · [Comparison](COMPARISON.md) · [AKAS](AKAS.md) · [Baselines](BASELINES.md) · [Measurements](measurements.csv)

Timings cover one warm localhost request through receipt of complete PCM16. Wall time and real-time factor (RTF, wall time divided by audio duration) are per-utterance medians. Comparison ratios use geometric means of matched successful utterances. Pass counts cover request completion, format and non-silent audio. AKAS also reports end of speech. WAV byte determinism is assessed within each runtime. Both runtimes used the same case-specific presets. Intelligibility, pronunciation and MOS were outside the measurement scope.

Model Zyphra/ZONOS3-Mini · q8_0 · 44,100 Hz · case-specific voices are listed in the overview.
Run [20261117113736893998-27a26a5b](../../runs/20261117113736893998-27a26a5b/run.json) · report SHA-256 `376fc70ccea49e4fd28cca8dfad0eeab473638125df92e4e76905d606b64bb76`.
Checkpoint SHA-256 `60568efc88e2f9231f4f5c061e246421f575986642d72ab9ae68f77d6cf76be4`. Projector SHA-256 `aeeb92325b4b6927bef656390b78e1246148065f947d8b59c2663557a809414d`.
Compiled program SHA-256 `5b60eb5f09a777dd4343bb989c7a52e8305adea3a3b4f0d2aa08ee65df4ca5c6`.
Compiled codec frame capacities: 16, 32, 64, 128, 256, 384.

| Language | Case | Tier | Status | Repeats | Wall ms | Audio s | RTF | PCM repeatable | Example | Error |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| en | en-0007 | short | ok | 2 | 634.1 | 2.467 | 0.257 | yes | [WAV](../../runs/20261117113736893998-27a26a5b/audio/01-01.wav) |  |
| en | en-0320 | short | ok | 2 | 668.4 | 2.733 | 0.245 | yes | [WAV](../../runs/20261117113736893998-27a26a5b/audio/02-01.wav) |  |
| en | en-0942 | medium | ok | 2 | 746.2 | 3.400 | 0.219 | yes | [WAV](../../runs/20261117113736893998-27a26a5b/audio/03-01.wav) |  |
| en | en-0030 | medium | ok | 2 | 940.0 | 4.400 | 0.214 | yes | [WAV](../../runs/20261117113736893998-27a26a5b/audio/04-01.wav) |  |
| en | en-0082 | long | ok | 2 | 1185.1 | 6.067 | 0.195 | yes | [WAV](../../runs/20261117113736893998-27a26a5b/audio/05-01.wav) |  |
| en | en-0977 | long | ok | 2 | 1288.0 | 6.467 | 0.199 | yes | [WAV](../../runs/20261117113736893998-27a26a5b/audio/06-01.wav) |  |
| fr | fr-0007 | short | ok | 2 | 1252.4 | 5.733 | 0.218 | yes | [WAV](../../runs/20261117113736893998-27a26a5b/audio/07-01.wav) |  |
| fr | fr-0320 | short | ok | 2 | 1428.2 | 6.267 | 0.228 | yes | [WAV](../../runs/20261117113736893998-27a26a5b/audio/08-01.wav) |  |
| fr | fr-0942 | medium | ok | 2 | 1510.0 | 6.933 | 0.218 | yes | [WAV](../../runs/20261117113736893998-27a26a5b/audio/09-01.wav) |  |
| fr | fr-0030 | medium | ok | 2 | 1959.3 | 9.133 | 0.215 | yes | [WAV](../../runs/20261117113736893998-27a26a5b/audio/10-01.wav) |  |
| fr | fr-0082 | long | ok | 2 | 2422.3 | 11.200 | 0.216 | yes | [WAV](../../runs/20261117113736893998-27a26a5b/audio/11-01.wav) |  |
| fr | fr-0977 | long | ok | 2 | 2629.0 | 11.200 | 0.235 | yes | [WAV](../../runs/20261117113736893998-27a26a5b/audio/12-01.wav) |  |
| es | es-0007 | short | ok | 2 | 701.5 | 2.267 | 0.309 | yes | [WAV](../../runs/20261117113736893998-27a26a5b/audio/13-01.wav) |  |
| es | es-0320 | short | ok | 2 | 988.8 | 3.800 | 0.260 | yes | [WAV](../../runs/20261117113736893998-27a26a5b/audio/14-01.wav) |  |
| es | es-0942 | medium | ok | 2 | 935.6 | 4.400 | 0.213 | yes | [WAV](../../runs/20261117113736893998-27a26a5b/audio/15-01.wav) |  |
| es | es-0030 | medium | ok | 2 | 1026.7 | 5.067 | 0.203 | yes | [WAV](../../runs/20261117113736893998-27a26a5b/audio/16-01.wav) |  |
| es | es-0082 | long | ok | 2 | 1423.7 | 7.733 | 0.184 | yes | [WAV](../../runs/20261117113736893998-27a26a5b/audio/17-01.wav) |  |
| es | es-0977 | long | ok | 2 | 1713.8 | 8.933 | 0.192 | yes | [WAV](../../runs/20261117113736893998-27a26a5b/audio/18-01.wav) |  |
| de | de-0007 | short | ok | 2 | 839.8 | 4.067 | 0.207 | yes | [WAV](../../runs/20261117113736893998-27a26a5b/audio/19-01.wav) |  |
| de | de-0320 | short | ok | 2 | 789.8 | 3.400 | 0.232 | yes | [WAV](../../runs/20261117113736893998-27a26a5b/audio/20-01.wav) |  |
| de | de-0942 | medium | ok | 2 | 939.5 | 4.267 | 0.220 | yes | [WAV](../../runs/20261117113736893998-27a26a5b/audio/21-01.wav) |  |
| de | de-0030 | medium | ok | 2 | 1042.2 | 4.667 | 0.223 | yes | [WAV](../../runs/20261117113736893998-27a26a5b/audio/22-01.wav) |  |
| de | de-0082 | long | ok | 2 | 1736.5 | 8.600 | 0.202 | yes | [WAV](../../runs/20261117113736893998-27a26a5b/audio/23-01.wav) |  |
| de | de-0977 | long | ok | 2 | 1897.6 | 8.667 | 0.219 | yes | [WAV](../../runs/20261117113736893998-27a26a5b/audio/24-01.wav) |  |
| it | it-0007 | short | ok | 2 | 839.4 | 3.200 | 0.262 | yes | [WAV](../../runs/20261117113736893998-27a26a5b/audio/25-01.wav) |  |
| it | it-0320 | short | ok | 2 | 1167.1 | 4.533 | 0.257 | yes | [WAV](../../runs/20261117113736893998-27a26a5b/audio/26-01.wav) |  |
| it | it-0942 | medium | ok | 2 | 1097.0 | 4.400 | 0.249 | yes | [WAV](../../runs/20261117113736893998-27a26a5b/audio/27-01.wav) |  |
| it | it-0030 | medium | ok | 2 | 1302.2 | 6.867 | 0.190 | yes | [WAV](../../runs/20261117113736893998-27a26a5b/audio/28-01.wav) |  |
| it | it-0082 | long | ok | 2 | 1405.7 | 7.867 | 0.179 | yes | [WAV](../../runs/20261117113736893998-27a26a5b/audio/29-01.wav) |  |
| it | it-0977 | long | ok | 2 | 1660.3 | 8.667 | 0.192 | yes | [WAV](../../runs/20261117113736893998-27a26a5b/audio/30-01.wav) |  |
| pt | pt-0007 | short | ok | 2 | 755.5 | 3.467 | 0.218 | yes | [WAV](../../runs/20261117113736893998-27a26a5b/audio/31-01.wav) |  |
| pt | pt-0320 | short | ok | 2 | 760.7 | 3.467 | 0.219 | yes | [WAV](../../runs/20261117113736893998-27a26a5b/audio/32-01.wav) |  |
| pt | pt-0942 | medium | ok | 2 | 940.6 | 4.267 | 0.220 | yes | [WAV](../../runs/20261117113736893998-27a26a5b/audio/33-01.wav) |  |
| pt | pt-0030 | medium | ok | 2 | 1096.6 | 5.333 | 0.206 | yes | [WAV](../../runs/20261117113736893998-27a26a5b/audio/34-01.wav) |  |
| pt | pt-0082 | long | ok | 2 | 1661.1 | 8.467 | 0.196 | yes | [WAV](../../runs/20261117113736893998-27a26a5b/audio/35-01.wav) |  |
| pt | pt-0977 | long | ok | 2 | 1793.6 | 9.267 | 0.194 | yes | [WAV](../../runs/20261117113736893998-27a26a5b/audio/36-01.wav) |  |
| zh | zh-0007 | short | ok | 2 | 697.2 | 2.800 | 0.249 | yes | [WAV](../../runs/20261117113736893998-27a26a5b/audio/37-01.wav) |  |
| zh | zh-0320 | short | ok | 2 | 826.3 | 3.867 | 0.214 | yes | [WAV](../../runs/20261117113736893998-27a26a5b/audio/38-01.wav) |  |
| zh | zh-0942 | medium | ok | 2 | 764.1 | 3.333 | 0.229 | yes | [WAV](../../runs/20261117113736893998-27a26a5b/audio/39-01.wav) |  |
| zh | zh-0030 | medium | ok | 2 | 798.3 | 3.467 | 0.230 | yes | [WAV](../../runs/20261117113736893998-27a26a5b/audio/40-01.wav) |  |
| zh | zh-0082 | long | ok | 2 | 1006.4 | 4.800 | 0.210 | yes | [WAV](../../runs/20261117113736893998-27a26a5b/audio/41-01.wav) |  |
| zh | zh-0977 | long | ok | 2 | 1300.5 | 6.067 | 0.214 | yes | [WAV](../../runs/20261117113736893998-27a26a5b/audio/42-01.wav) |  |
| ko | ko-0007 | short | ok | 2 | 1041.2 | 4.000 | 0.260 | yes | [WAV](../../runs/20261117113736893998-27a26a5b/audio/43-01.wav) |  |
| ko | ko-0320 | short | ok | 2 | 1100.4 | 4.267 | 0.258 | yes | [WAV](../../runs/20261117113736893998-27a26a5b/audio/44-01.wav) |  |
| ko | ko-0942 | medium | ok | 2 | 927.5 | 3.400 | 0.273 | yes | [WAV](../../runs/20261117113736893998-27a26a5b/audio/45-01.wav) |  |
| ko | ko-0030 | medium | ok | 2 | 475.4 | 1.067 | 0.446 | yes | [WAV](../../runs/20261117113736893998-27a26a5b/audio/46-01.wav) |  |
| ko | ko-0082 | long | ok | 2 | 1561.7 | 6.600 | 0.237 | yes | [WAV](../../runs/20261117113736893998-27a26a5b/audio/47-01.wav) |  |
| ko | ko-0977 | long | ok | 2 | 1680.4 | 8.400 | 0.200 | yes | [WAV](../../runs/20261117113736893998-27a26a5b/audio/48-01.wav) |  |
| hi | hi-0007 | short | ok | 2 | 1338.5 | 5.533 | 0.242 | yes | [WAV](../../runs/20261117113736893998-27a26a5b/audio/49-01.wav) |  |
| hi | hi-0320 | short | ok | 2 | 1010.1 | 4.067 | 0.248 | yes | [WAV](../../runs/20261117113736893998-27a26a5b/audio/50-01.wav) |  |
| hi | hi-0942 | medium | ok | 2 | 940.8 | 3.733 | 0.252 | yes | [WAV](../../runs/20261117113736893998-27a26a5b/audio/51-01.wav) |  |
| hi | hi-0030 | medium | ok | 2 | 1277.3 | 5.400 | 0.237 | yes | [WAV](../../runs/20261117113736893998-27a26a5b/audio/52-01.wav) |  |
| hi | hi-0082 | long | ok | 2 | 2127.1 | 8.600 | 0.247 | yes | [WAV](../../runs/20261117113736893998-27a26a5b/audio/53-01.wav) |  |
| hi | hi-0977 | long | ok | 2 | 1855.7 | 7.667 | 0.242 | yes | [WAV](../../runs/20261117113736893998-27a26a5b/audio/54-01.wav) |  |
| te | te-0007 | short | ok | 2 | 947.2 | 3.667 | 0.258 | yes | [WAV](../../runs/20261117113736893998-27a26a5b/audio/55-01.wav) |  |
| te | te-0320 | short | ok | 2 | 925.1 | 4.133 | 0.224 | yes | [WAV](../../runs/20261117113736893998-27a26a5b/audio/56-01.wav) |  |
| te | te-0942 | medium | ok | 2 | 774.2 | 3.467 | 0.223 | yes | [WAV](../../runs/20261117113736893998-27a26a5b/audio/57-01.wav) |  |
| te | te-0030 | medium | ok | 2 | 1383.8 | 6.467 | 0.214 | yes | [WAV](../../runs/20261117113736893998-27a26a5b/audio/58-01.wav) |  |
| te | te-0082 | long | ok | 2 | 1492.7 | 7.200 | 0.207 | yes | [WAV](../../runs/20261117113736893998-27a26a5b/audio/59-01.wav) |  |
| te | te-0977 | long | ok | 2 | 2122.8 | 10.733 | 0.198 | yes | [WAV](../../runs/20261117113736893998-27a26a5b/audio/60-01.wav) |  |
