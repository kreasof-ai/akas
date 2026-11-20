# Qwen3 ASR benchmark

[Overview](README.md) · [Comparison](COMPARISON.md) · [AKAS](AKAS.md) · [Baselines](BASELINES.md) · [Measurements](measurements.csv)

Workload `asr-length-v1` uses 9 fixed clips. Protocol `warm-asr-request-v1`.

Warm wall time covers the request through receipt of the complete transcript over local process pipes. The timer includes WAV loading, feature extraction and greedy decoding. Model loading and context reset are outside the timer. RTF is wall seconds per second of input audio. WER and CER use case-folded NFKC text with punctuation removed. The clips are recorded speech from LibriSpeech dev-clean, validation subset, reader 1272. Both profiles are unqualified for independent full-model accuracy.

| Runtime | GPU | Complete and deterministic cases | Run |
| --- | --- | ---: | --- |
| AKAS (webgpu) | AMD Radeon RX 6700 XT | 9/9 | [20261120084341194215-d25a4c19](../../runs/20261120084341194215-d25a4c19/run.json) |
| llama.cpp (vulkan) | AMD Radeon RX 6700 XT (12272 MiB, 11474 MiB free) | 9/9 | [20261120071231701250-5ba2ac25](../../runs/20261120071231701250-5ba2ac25/run.json) |
