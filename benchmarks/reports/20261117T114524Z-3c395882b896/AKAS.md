# akas speech measurements

[Overview](README.md) · [Comparison](COMPARISON.md) · [AKAS](AKAS.md) · [Baselines](BASELINES.md) · [Measurements](measurements.csv)

Timings cover one warm localhost request through receipt of complete PCM16. Wall time and real-time factor (RTF, wall time divided by audio duration) are per-utterance medians. Comparison ratios use geometric means of matched successful utterances. Pass counts cover request completion, format and non-silent audio. AKAS also reports end of speech. WAV byte determinism is assessed within each runtime. Both runtimes used the same case-specific presets. Intelligibility, pronunciation and MOS were outside the measurement scope.

Model Zyphra/ZONOS3-Mini · q8_0 · 44,100 Hz · case-specific voices are listed in the overview.
Run [20261117114524389596-212345e0](https://github.com/kreasof-ai/akas/blob/de95b36413a35e8f59d96945325fb008c11d18bf/benchmarks/runs/20261117114524389596-212345e0/run.json) · report SHA-256 `f28bcc23a57d46ee2d9e85699a0dc6aaa61ad1a2ff444dad20933e5f6b048a2f`.
Checkpoint SHA-256 `60568efc88e2f9231f4f5c061e246421f575986642d72ab9ae68f77d6cf76be4`. Projector SHA-256 `aeeb92325b4b6927bef656390b78e1246148065f947d8b59c2663557a809414d`.
Compiled program SHA-256 `206ccd11064c0c41be53c98e9e2d33ebecdc9b19869841f138473e5fa255ea6f`.
Compiled codec frame capacities: 16, 32, 64, 128, 256, 384.

| Language | Case | Tier | Status | Repeats | Wall ms | Audio s | RTF | PCM repeatable | Example | Error |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| en | en-0007 | short | ok | 2 | 670.5 | 2.733 | 0.245 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/de95b36413a35e8f59d96945325fb008c11d18bf/benchmarks/runs/20261117114524389596-212345e0/audio/01-01.wav) |  |
| en | en-0320 | short | ok | 2 | 649.2 | 2.600 | 0.250 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/de95b36413a35e8f59d96945325fb008c11d18bf/benchmarks/runs/20261117114524389596-212345e0/audio/02-01.wav) |  |
| en | en-0942 | medium | ok | 2 | 802.1 | 3.400 | 0.236 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/de95b36413a35e8f59d96945325fb008c11d18bf/benchmarks/runs/20261117114524389596-212345e0/audio/03-01.wav) |  |
| en | en-0030 | medium | ok | 2 | 956.2 | 4.400 | 0.217 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/de95b36413a35e8f59d96945325fb008c11d18bf/benchmarks/runs/20261117114524389596-212345e0/audio/04-01.wav) |  |
| en | en-0082 | long | ok | 2 | 1193.7 | 5.733 | 0.208 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/de95b36413a35e8f59d96945325fb008c11d18bf/benchmarks/runs/20261117114524389596-212345e0/audio/05-01.wav) |  |
| en | en-0977 | long | ok | 2 | 1235.5 | 6.333 | 0.195 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/de95b36413a35e8f59d96945325fb008c11d18bf/benchmarks/runs/20261117114524389596-212345e0/audio/06-01.wav) |  |
| fr | fr-0007 | short | ok | 2 | 1159.5 | 4.800 | 0.242 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/de95b36413a35e8f59d96945325fb008c11d18bf/benchmarks/runs/20261117114524389596-212345e0/audio/07-01.wav) |  |
| fr | fr-0320 | short | ok | 2 | 1290.8 | 5.867 | 0.220 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/de95b36413a35e8f59d96945325fb008c11d18bf/benchmarks/runs/20261117114524389596-212345e0/audio/08-01.wav) |  |
| fr | fr-0942 | medium | ok | 2 | 1421.9 | 6.467 | 0.220 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/de95b36413a35e8f59d96945325fb008c11d18bf/benchmarks/runs/20261117114524389596-212345e0/audio/09-01.wav) |  |
| fr | fr-0030 | medium | ok | 2 | 1654.3 | 8.133 | 0.203 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/de95b36413a35e8f59d96945325fb008c11d18bf/benchmarks/runs/20261117114524389596-212345e0/audio/10-01.wav) |  |
| fr | fr-0082 | long | ok | 2 | 2160.2 | 10.667 | 0.203 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/de95b36413a35e8f59d96945325fb008c11d18bf/benchmarks/runs/20261117114524389596-212345e0/audio/11-01.wav) |  |
| fr | fr-0977 | long | ok | 2 | 2506.2 | 10.667 | 0.235 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/de95b36413a35e8f59d96945325fb008c11d18bf/benchmarks/runs/20261117114524389596-212345e0/audio/12-01.wav) |  |
| es | es-0007 | short | ok | 2 | 708.6 | 2.333 | 0.304 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/de95b36413a35e8f59d96945325fb008c11d18bf/benchmarks/runs/20261117114524389596-212345e0/audio/13-01.wav) |  |
| es | es-0320 | short | ok | 2 | 914.8 | 3.400 | 0.269 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/de95b36413a35e8f59d96945325fb008c11d18bf/benchmarks/runs/20261117114524389596-212345e0/audio/14-01.wav) |  |
| es | es-0942 | medium | ok | 2 | 1135.4 | 4.533 | 0.250 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/de95b36413a35e8f59d96945325fb008c11d18bf/benchmarks/runs/20261117114524389596-212345e0/audio/15-01.wav) |  |
| es | es-0030 | medium | ok | 2 | 1097.7 | 5.533 | 0.198 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/de95b36413a35e8f59d96945325fb008c11d18bf/benchmarks/runs/20261117114524389596-212345e0/audio/16-01.wav) |  |
| es | es-0082 | long | ok | 2 | 1381.9 | 7.200 | 0.192 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/de95b36413a35e8f59d96945325fb008c11d18bf/benchmarks/runs/20261117114524389596-212345e0/audio/17-01.wav) |  |
| es | es-0977 | long | ok | 2 | 1833.2 | 9.667 | 0.190 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/de95b36413a35e8f59d96945325fb008c11d18bf/benchmarks/runs/20261117114524389596-212345e0/audio/18-01.wav) |  |
| de | de-0007 | short | ok | 2 | 862.2 | 3.867 | 0.223 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/de95b36413a35e8f59d96945325fb008c11d18bf/benchmarks/runs/20261117114524389596-212345e0/audio/19-01.wav) |  |
| de | de-0320 | short | ok | 2 | 775.4 | 3.533 | 0.219 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/de95b36413a35e8f59d96945325fb008c11d18bf/benchmarks/runs/20261117114524389596-212345e0/audio/20-01.wav) |  |
| de | de-0942 | medium | ok | 2 | 958.2 | 4.267 | 0.225 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/de95b36413a35e8f59d96945325fb008c11d18bf/benchmarks/runs/20261117114524389596-212345e0/audio/21-01.wav) |  |
| de | de-0030 | medium | ok | 2 | 976.2 | 4.467 | 0.219 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/de95b36413a35e8f59d96945325fb008c11d18bf/benchmarks/runs/20261117114524389596-212345e0/audio/22-01.wav) |  |
| de | de-0082 | long | ok | 2 | 1720.9 | 8.933 | 0.193 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/de95b36413a35e8f59d96945325fb008c11d18bf/benchmarks/runs/20261117114524389596-212345e0/audio/23-01.wav) |  |
| de | de-0977 | long | ok | 2 | 1528.0 | 8.133 | 0.188 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/de95b36413a35e8f59d96945325fb008c11d18bf/benchmarks/runs/20261117114524389596-212345e0/audio/24-01.wav) |  |
| it | it-0007 | short | ok | 2 | 722.6 | 3.133 | 0.231 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/de95b36413a35e8f59d96945325fb008c11d18bf/benchmarks/runs/20261117114524389596-212345e0/audio/25-01.wav) |  |
| it | it-0320 | short | ok | 2 | 990.5 | 4.533 | 0.218 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/de95b36413a35e8f59d96945325fb008c11d18bf/benchmarks/runs/20261117114524389596-212345e0/audio/26-01.wav) |  |
| it | it-0942 | medium | ok | 2 | 961.9 | 4.400 | 0.219 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/de95b36413a35e8f59d96945325fb008c11d18bf/benchmarks/runs/20261117114524389596-212345e0/audio/27-01.wav) |  |
| it | it-0030 | medium | ok | 2 | 1407.1 | 6.867 | 0.205 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/de95b36413a35e8f59d96945325fb008c11d18bf/benchmarks/runs/20261117114524389596-212345e0/audio/28-01.wav) |  |
| it | it-0082 | long | ok | 2 | 1760.4 | 8.400 | 0.210 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/de95b36413a35e8f59d96945325fb008c11d18bf/benchmarks/runs/20261117114524389596-212345e0/audio/29-01.wav) |  |
| it | it-0977 | long | ok | 2 | 1823.3 | 8.667 | 0.210 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/de95b36413a35e8f59d96945325fb008c11d18bf/benchmarks/runs/20261117114524389596-212345e0/audio/30-01.wav) |  |
| pt | pt-0007 | short | ok | 2 | 842.2 | 3.667 | 0.230 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/de95b36413a35e8f59d96945325fb008c11d18bf/benchmarks/runs/20261117114524389596-212345e0/audio/31-01.wav) |  |
| pt | pt-0320 | short | ok | 2 | 909.2 | 3.800 | 0.239 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/de95b36413a35e8f59d96945325fb008c11d18bf/benchmarks/runs/20261117114524389596-212345e0/audio/32-01.wav) |  |
| pt | pt-0942 | medium | ok | 2 | 1099.9 | 4.400 | 0.250 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/de95b36413a35e8f59d96945325fb008c11d18bf/benchmarks/runs/20261117114524389596-212345e0/audio/33-01.wav) |  |
| pt | pt-0030 | medium | ok | 2 | 1283.2 | 5.533 | 0.232 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/de95b36413a35e8f59d96945325fb008c11d18bf/benchmarks/runs/20261117114524389596-212345e0/audio/34-01.wav) |  |
| pt | pt-0082 | long | ok | 2 | 1600.6 | 8.133 | 0.197 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/de95b36413a35e8f59d96945325fb008c11d18bf/benchmarks/runs/20261117114524389596-212345e0/audio/35-01.wav) |  |
| pt | pt-0977 | long | ok | 2 | 1808.3 | 8.467 | 0.214 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/de95b36413a35e8f59d96945325fb008c11d18bf/benchmarks/runs/20261117114524389596-212345e0/audio/36-01.wav) |  |
| zh | zh-0007 | short | ok | 2 | 718.2 | 2.933 | 0.245 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/de95b36413a35e8f59d96945325fb008c11d18bf/benchmarks/runs/20261117114524389596-212345e0/audio/37-01.wav) |  |
| zh | zh-0320 | short | ok | 2 | 840.4 | 3.400 | 0.247 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/de95b36413a35e8f59d96945325fb008c11d18bf/benchmarks/runs/20261117114524389596-212345e0/audio/38-01.wav) |  |
| zh | zh-0942 | medium | ok | 2 | 787.9 | 3.267 | 0.241 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/de95b36413a35e8f59d96945325fb008c11d18bf/benchmarks/runs/20261117114524389596-212345e0/audio/39-01.wav) |  |
| zh | zh-0030 | medium | ok | 2 | 815.9 | 3.467 | 0.235 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/de95b36413a35e8f59d96945325fb008c11d18bf/benchmarks/runs/20261117114524389596-212345e0/audio/40-01.wav) |  |
| zh | zh-0082 | long | ok | 2 | 1050.0 | 4.867 | 0.216 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/de95b36413a35e8f59d96945325fb008c11d18bf/benchmarks/runs/20261117114524389596-212345e0/audio/41-01.wav) |  |
| zh | zh-0977 | long | ok | 2 | 1095.0 | 5.200 | 0.211 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/de95b36413a35e8f59d96945325fb008c11d18bf/benchmarks/runs/20261117114524389596-212345e0/audio/42-01.wav) |  |
| ko | ko-0007 | short | ok | 2 | 827.6 | 3.867 | 0.214 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/de95b36413a35e8f59d96945325fb008c11d18bf/benchmarks/runs/20261117114524389596-212345e0/audio/43-01.wav) |  |
| ko | ko-0320 | short | ok | 2 | 1000.4 | 4.400 | 0.227 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/de95b36413a35e8f59d96945325fb008c11d18bf/benchmarks/runs/20261117114524389596-212345e0/audio/44-01.wav) |  |
| ko | ko-0942 | medium | ok | 2 | 774.5 | 3.400 | 0.228 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/de95b36413a35e8f59d96945325fb008c11d18bf/benchmarks/runs/20261117114524389596-212345e0/audio/45-01.wav) |  |
| ko | ko-0030 | medium | ok | 2 | 1212.0 | 6.067 | 0.200 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/de95b36413a35e8f59d96945325fb008c11d18bf/benchmarks/runs/20261117114524389596-212345e0/audio/46-01.wav) |  |
| ko | ko-0082 | long | ok | 2 | 1203.2 | 5.667 | 0.212 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/de95b36413a35e8f59d96945325fb008c11d18bf/benchmarks/runs/20261117114524389596-212345e0/audio/47-01.wav) |  |
| ko | ko-0977 | long | ok | 2 | 1978.9 | 9.867 | 0.201 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/de95b36413a35e8f59d96945325fb008c11d18bf/benchmarks/runs/20261117114524389596-212345e0/audio/48-01.wav) |  |
| hi | hi-0007 | short | ok | 2 | 1225.0 | 5.067 | 0.242 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/de95b36413a35e8f59d96945325fb008c11d18bf/benchmarks/runs/20261117114524389596-212345e0/audio/49-01.wav) |  |
| hi | hi-0320 | short | ok | 2 | 816.0 | 3.667 | 0.223 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/de95b36413a35e8f59d96945325fb008c11d18bf/benchmarks/runs/20261117114524389596-212345e0/audio/50-01.wav) |  |
| hi | hi-0942 | medium | ok | 2 | 824.6 | 3.800 | 0.217 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/de95b36413a35e8f59d96945325fb008c11d18bf/benchmarks/runs/20261117114524389596-212345e0/audio/51-01.wav) |  |
| hi | hi-0030 | medium | ok | 2 | 1097.7 | 5.067 | 0.217 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/de95b36413a35e8f59d96945325fb008c11d18bf/benchmarks/runs/20261117114524389596-212345e0/audio/52-01.wav) |  |
| hi | hi-0082 | long | ok | 2 | 1626.9 | 7.267 | 0.224 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/de95b36413a35e8f59d96945325fb008c11d18bf/benchmarks/runs/20261117114524389596-212345e0/audio/53-01.wav) |  |
| hi | hi-0977 | long | ok | 2 | 1851.9 | 7.800 | 0.237 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/de95b36413a35e8f59d96945325fb008c11d18bf/benchmarks/runs/20261117114524389596-212345e0/audio/54-01.wav) |  |
| te | te-0007 | short | ok | 2 | 947.7 | 3.667 | 0.258 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/de95b36413a35e8f59d96945325fb008c11d18bf/benchmarks/runs/20261117114524389596-212345e0/audio/55-01.wav) |  |
| te | te-0320 | short | ok | 2 | 948.2 | 3.733 | 0.254 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/de95b36413a35e8f59d96945325fb008c11d18bf/benchmarks/runs/20261117114524389596-212345e0/audio/56-01.wav) |  |
| te | te-0942 | medium | ok | 2 | 1005.6 | 3.800 | 0.265 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/de95b36413a35e8f59d96945325fb008c11d18bf/benchmarks/runs/20261117114524389596-212345e0/audio/57-01.wav) |  |
| te | te-0030 | medium | ok | 2 | 1748.1 | 7.200 | 0.243 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/de95b36413a35e8f59d96945325fb008c11d18bf/benchmarks/runs/20261117114524389596-212345e0/audio/58-01.wav) |  |
| te | te-0082 | long | ok | 2 | 1698.5 | 6.933 | 0.245 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/de95b36413a35e8f59d96945325fb008c11d18bf/benchmarks/runs/20261117114524389596-212345e0/audio/59-01.wav) |  |
| te | te-0977 | long | ok | 2 | 2205.6 | 11.800 | 0.187 | yes | [WAV](https://github.com/kreasof-ai/akas/blob/de95b36413a35e8f59d96945325fb008c11d18bf/benchmarks/runs/20261117114524389596-212345e0/audio/60-01.wav) |  |
