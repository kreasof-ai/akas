# Native BF16 streaming baseline

These are Zyphra native CUDA **baseline diagnostics**, not qualified AKAS measurements. The GGUF/llama.cpp path is not part of this comparison.

Model: **Zyphra/ZONOS3-Mini**; weights: **BF16 LM** with the FP32 `keep2_plain` codec. Fixed English input: **30 words / 177 UTF-8 bytes**. Each request uses **50 prompt frames** (47 model-input frames + 3 silence-prefill frames), up to **384 generated tokens** within a **512-frame** limit.

Flat admission up to the requested request limit; no request-class bucketing. Prefix streaming is enabled. The scheduler fix prevents request-slot recycling while earlier decodes remain active. C64 had the highest measured audio throughput; C256 is a successful capacity result, not a latency target.

| Concurrency | Valid audio | Wall time (s) | Audio seconds/s | Peak GPU (GiB) | Evidence |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 1/1 | 0.70 | 15.53 | 4.83 | [20261117161834374360-9fc8577b](https://github.com/kreasof-ai/akas/blob/68ec4fd465c04f4552f8979a2204b601228f780b/benchmarks/runs/20261117161834374360-9fc8577b/run.json) |
| 4 | 4/4 | 1.06 | 42.46 | 4.79 | [20261117160518967887-a44525a6](https://github.com/kreasof-ai/akas/blob/68ec4fd465c04f4552f8979a2204b601228f780b/benchmarks/runs/20261117160518967887-a44525a6/run.json) |
| 16 | 16/16 | 3.60 | 47.81 | 5.14 | [20261117161834374360-9fc8577b](https://github.com/kreasof-ai/akas/blob/68ec4fd465c04f4552f8979a2204b601228f780b/benchmarks/runs/20261117161834374360-9fc8577b/run.json) |
| 64 | 64/64 | 10.45 | 68.12 | 6.49 | [20261117160519883604-b75f11cd](https://github.com/kreasof-ai/akas/blob/68ec4fd465c04f4552f8979a2204b601228f780b/benchmarks/runs/20261117160519883604-b75f11cd/run.json) |
| 256 | 256/256 | 163.58 | 17.35 | 7.48 | [20261117160602268099-ed904351](https://github.com/kreasof-ai/akas/blob/68ec4fd465c04f4552f8979a2204b601228f780b/benchmarks/runs/20261117160602268099-ed904351/run.json) |

## Source and weight packaging

For the native baseline, the full training generator can be replaced by a **compact, dtype-preserving codec checkpoint** with identical output. The matrix used the original assets; the separately qualified compact package is linked below. The LM remains BF16; no weights are quantized for size. The local scheduler and loading changes are retained in each run's **`dependencies/<dependency>/source.tar.gz`** artifact. Inspect the linked run's `run.json` and dependency snapshot for the exact source used; do not use a mutable checkout as provenance.

Codec packaging: **0.577 GB**, replacing **8.48 GB** of full generator + override assets. With the 1.772 GB BF16 LM, the required native model data is approximately **2.35 GB** instead of **10.25 GB**.

Bitwise codec qualification: [1-, 7-, and 32-frame identical-output checks](https://github.com/kreasof-ai/akas/blob/68ec4fd465c04f4552f8979a2204b601228f780b/benchmarks/runs/20261117162500830863-7586a1cb/qualification.json). Compact C1/C16 HTTP integration: [20261117162440472501-cbda6b06](https://github.com/kreasof-ai/akas/blob/68ec4fd465c04f4552f8979a2204b601228f780b/benchmarks/runs/20261117162440472501-cbda6b06/run.json).

The compact checkpoint remains a local build artifact; it is intentionally not uploaded to external storage. Its size, format and SHA-256 are retained in the qualification record.

Exact baseline source bundle: [dependencies/zonos3/source.tar.gz](https://github.com/kreasof-ai/akas/blob/68ec4fd465c04f4552f8979a2204b601228f780b/benchmarks/runs/20261117163259599695-66d7e9b8/dependencies/zonos3/source.tar.gz). This supplemental bundle includes `model/` and `servers/` and is the authoritative source for the compact codec loader and safe slot-lifecycle scheduler change.

Run artifacts, source snapshots, exact dependency revisions, per-request audio hashes and server logs are retained under the normal benchmark archive. These measurements establish native baseline functionality and capacity; they do not claim AKAS BF16 correctness or qualified speedups.
