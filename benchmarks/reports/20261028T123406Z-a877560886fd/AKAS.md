# AKAS configurations

[Campaign overview](README.md) · [Side-by-side tables](COMPARISON.md) · [CSV](measurements.csv)

Decode rates are medians of five samples after two warmups. One request generates 128 outputs; 127 decode steps are timed. Both timers cover completed wall decode including greedy sampling and control dispatch; model load, real prefill, snapshot restore and validation are excluded. AKAS uses GPU token feedback; llama.cpp b11946 Vulkan C API uses its stock greedy sampler. Both restore complete real-prefill recurrent/KV state and the initial prediction. Prompts use fixed synthetic token IDs, never fabricated KV. Prefill observations are single samples in the portable data, not repeated benchmarks, and do not gate decode parity. A dash means no measured median. The maximum-prefix AKAS state is imported from sealed real-prefill run 20261028130402880342-2c41aef6 with checked hashes and unchanged prefill arithmetic/storage. Short prefixes through 8,192 select the 32-partition program; longer ones select the shared-KV 512-partition program. Both reserve a 131,072-token context. Generated sequences differ at some prefixes; performance parity does not establish identical text or end-to-end request latency.

<a id="lfm3-1.7b-q4_0-webgpu-float16-ar"></a>
## lfm3-1.7b / akas / q4_0 / webgpu / float16 / ar

Profile status: **passed**; qualification: **passed**; effective KV dtype: `float16`. Evidence: [20261028142101939494-891196a6](https://github.com/akbar2habibullah/akas/blob/c1d6d85ad4ddabdfce0d2d0e4de2a7a3de6cf5f9/benchmarks/runs/20261028142101939494-891196a6/run.json).

Peak device GiB: —; peak host process RSS GiB: —. Loading and in-process compilation are included in host RSS.

Selected recipe: `{"chunk": 512, "graphs": true, "prefill_requests": 1, "trials": 1}`.

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | real-prefill-state-restored-wall-decode-v1 | — | 247.5 | — | — | — | — | — | 5 | passed | [20261028143228923276-636cfb77](https://github.com/akbar2habibullah/akas/blob/c1d6d85ad4ddabdfce0d2d0e4de2a7a3de6cf5f9/benchmarks/runs/20261028143228923276-636cfb77/run.json) |
| 1 | 512 | 128 | real-prefill-state-restored-wall-decode-v1 | — | 244.1 | — | — | — | — | — | 5 | passed | [20261028143228923276-636cfb77](https://github.com/akbar2habibullah/akas/blob/c1d6d85ad4ddabdfce0d2d0e4de2a7a3de6cf5f9/benchmarks/runs/20261028143228923276-636cfb77/run.json) |
| 1 | 2048 | 128 | real-prefill-state-restored-wall-decode-v1 | — | 225.4 | — | — | — | — | — | 5 | passed | [20261028143228923276-636cfb77](https://github.com/akbar2habibullah/akas/blob/c1d6d85ad4ddabdfce0d2d0e4de2a7a3de6cf5f9/benchmarks/runs/20261028143228923276-636cfb77/run.json) |
| 1 | 8192 | 128 | real-prefill-state-restored-wall-decode-v1 | — | 186.6 | — | — | — | — | — | 5 | passed | [20261028143228923276-636cfb77](https://github.com/akbar2habibullah/akas/blob/c1d6d85ad4ddabdfce0d2d0e4de2a7a3de6cf5f9/benchmarks/runs/20261028143228923276-636cfb77/run.json) |
| 1 | 16384 | 128 | real-prefill-state-restored-wall-decode-v1 | — | 179.6 | — | — | — | — | — | 5 | passed | [20261028143228923276-636cfb77](https://github.com/akbar2habibullah/akas/blob/c1d6d85ad4ddabdfce0d2d0e4de2a7a3de6cf5f9/benchmarks/runs/20261028143228923276-636cfb77/run.json) |
| 1 | 32768 | 128 | real-prefill-state-restored-wall-decode-v1 | — | 153.4 | — | — | — | — | — | 5 | passed | [20261028143228923276-636cfb77](https://github.com/akbar2habibullah/akas/blob/c1d6d85ad4ddabdfce0d2d0e4de2a7a3de6cf5f9/benchmarks/runs/20261028143228923276-636cfb77/run.json) |
| 1 | 65536 | 128 | real-prefill-state-restored-wall-decode-v1 | — | 117.3 | — | — | — | — | — | 5 | passed | [20261028143228923276-636cfb77](https://github.com/akbar2habibullah/akas/blob/c1d6d85ad4ddabdfce0d2d0e4de2a7a3de6cf5f9/benchmarks/runs/20261028143228923276-636cfb77/run.json) |
| 1 | 130944 | 128 | real-prefill-state-restored-wall-decode-v1 | — | 79.5 | — | — | — | — | — | 5 | passed | [20261028142101939494-891196a6](https://github.com/akbar2habibullah/akas/blob/c1d6d85ad4ddabdfce0d2d0e4de2a7a3de6cf5f9/benchmarks/runs/20261028142101939494-891196a6/run.json) |
