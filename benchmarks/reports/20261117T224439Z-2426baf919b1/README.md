# ZONOS3-Mini: native BF16 streaming on NVIDIA L40S

Native BF16 LM weights, FP32 `keep2_plain`, and 44.1 kHz float32 PCM on CUDA. Both runtimes measured one HTTP prefix-streaming burst at each concurrency with the fixed 30-word (177-byte) English prompt, 50 prompt frames, and `max_tokens=384`.

[Three-panel figure](https://github.com/kreasof-ai/akas/blob/45bc9c006efd1a00dfb76df133ab3b1dfb348e5a/benchmarks/runs/20261117231016345646-b1f74c89/figures/throughput.svg) · [AKAS measurements](https://github.com/kreasof-ai/akas/blob/092e255f32f6bd5cb3f8e03fd1e8bc95802e033c/benchmarks/runs/20261117224439048177-b767426d/report.json) · [Zyphra native baseline](https://github.com/kreasof-ai/akas/blob/1042dbbe1dee7c0e6ad519374ecfd8ef85183f65/benchmarks/reports/20261117T160518Z-59b3379b9d08/README.md) · [Correctness probes](https://github.com/kreasof-ai/akas/blob/72bb2dc8a1fee29e3cfe2f1a40e967e6228c58f5/benchmarks/runs/20261117224224262017-91925158/report.json)

| Concurrency | Valid AKAS/native | AKAS wall s | Native wall s | AKAS audio s/s | Native audio s/s | Speedup | AKAS/native median TTFA (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1/1 | 0.41 | 0.70 | 28.80 | 15.53 | 1.85× | 43/150 |
| 4 | 4/4 | 0.84 | 1.06 | 52.10 | 42.46 | 1.23× | 20/60 |
| 16 | 16/16 | 2.59 | 3.60 | 68.73 | 47.81 | 1.44× | 51/722 |
| 64 | 64/64 | 6.17 | 10.45 | 115.60 | 68.12 | 1.70× | 287/1708 |
| 256 | 256/256 | 20.15 | 163.58 | 142.11 | 17.35 | 8.19× | 911/79026 |

Throughput is aggregate generated audio duration divided by burst wall time. TTFA is client time to first audio. Compilation and model loading are excluded.

AKAS at C256 delivered **2.09×** Zyphra's best measured throughput (C64). Zyphra's C256 prefill budget serialized admission into waves; the matched C256 speedup includes the effect of that scheduling limit.

## Correctness and provenance

The measured CUDA bundle passed 34 codec probes against Zyphra's `ISTFTStream` (worst RMS 1.52e-04, max absolute 6.79e-03). Greedy and seeded sampled requests reproduced token rows and PCM bitwise on slot reuse across 682 replayed utterances. These are workload checks, not independent end-to-end equivalence or an intelligibility assessment.

CUDA bundle SHA-256: `02489ec2dbafff47cca1cf8aff9ad9959e57a5fd0ddf3f549813e9c12338c183`. The linked run bundles preserve source snapshots, dependency revisions, and phase manifests. The compact codec checkpoint is not part of the archive.
