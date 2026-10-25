# AKAS configurations

[Campaign overview](README.md) · [Side-by-side tables](COMPARISON.md) · [CSV](measurements.csv)

All throughput and displayed latency values are medians over the retained repetitions. AKAS native decode counts the outputs after the first token; burst throughput counts all outputs. AKAS burst timing includes loopback HTTP/SSE, vLLM and SGLang burst timing uses native engine calls, and llama.cpp burst timing uses concurrent HTTP requests. Phase rates have distinct protocols and cannot be compared as equivalent kernel timings. Weight labels describe source storage; execution precision is recorded per configuration and can differ between operations. Unqualified timing values remain visible with † and are excluded from qualified claims. — means no retained measurement, never zero throughput.

<a id="lfm2.5-24b-a2b-q4_0-cuda-float16-dspark"></a>
## lfm2.5-24b-a2b / akas / q4_0 / cuda / float16 / dspark

Profile status: **complete**; qualification: **passed**; effective KV dtype: `float16`. Evidence: [20261023105846475312-580635c0](../../runs/20261023105846475312-580635c0/run.json).

Peak device GiB: —; peak host process RSS GiB: 2.4. Loading and in-process compilation are included in host RSS.

Selected recipe: `{"chunk": 4096, "deferred": false, "graphs": true, "head": "akas.models.dspark.proposer:load", "head_options": {"bundle": "/home/sagemaker-user/akas/build/lfm-moe/dspark-long-qualification/draft", "checkpoint": "/home/sagemaker-user/akas/build/checkpoints/lfm2.5-24b-a2b/dspark-f16.gguf"}, "id": "2ab52555bcb84326", "mode": "dspark", "model_options": {"auxiliary_layers": [2, 10, 18, 26, 34], "kv_dtype": "float16"}, "policy": {"attention": {"decoded_kv": false, "native_tail": true, "splits": 64}, "moe": {"fused_swiglu": true}}, "prefill_requests": 1, "provider": "cuda", "round_capture": false, "target": "sm_89", "trials": 1, "window": 10}`.

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | native-token-step-v2 | 602.1 | 165.1 | 155.6 | 155.8 | 155.2 | 52.3 | 6.1 | 3 | passed | [20261023105846475312-580635c0](../../runs/20261023105846475312-580635c0/run.json) |
| 1 | 8192 | 128 | native-token-step-v2 | 4,257.3 | 176.9 | 48.4 | 48.3 | 48.5 | 1,895.9 | 5.9 | 3 | passed | [20261023105846475312-580635c0](../../runs/20261023105846475312-580635c0/run.json) |
| 1 | 130944 | 128 | native-token-step-v2 | 2,721.9 | 12.8 | 2.2 | 2.2 | 2.2 | 48,509.0 | 79.2 | 3 | passed | [20261023105846475312-580635c0](../../runs/20261023105846475312-580635c0/run.json) |
