# Speech benchmark report

[Overview](README.md) · [Comparison](COMPARISON.md) · [AKAS](AKAS.md) · [Baselines](BASELINES.md) · [Measurements](measurements.csv)

Workload `speech-multilingual-v1` covers **43 languages**, **258 utterances** and 2 measured requests per utterance after 1 warmup.

Protocol `client-speech-pcm-v1`. UTC 2026-11-17T07:17:44.218931Z to 2026-11-17T07:35:04.090784Z.

Timings cover one warm localhost request through receipt of complete PCM16. Wall time and real-time factor (RTF, wall time divided by audio duration) are per-utterance medians. Comparison ratios use geometric means of matched successful utterances. Pass counts cover request completion, format and non-silent audio. AKAS also reports end of speech. WAV byte determinism is assessed within each runtime. Both runtimes used one fixed voice for all languages. Intelligibility, pronunciation and MOS were outside the measurement scope.

Language coverage follows the [model card](https://huggingface.co/Zyphra/ZONOS3-Mini#languages). Prompts are from [FLORES-200 devtest](https://github.com/facebookresearch/flores/tree/main/flores200), licensed [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Six aligned one-based rows, two per length tier by English character count. Rows use translated text as distributed and exclude digits across the selected languages.

| Runtime | Device | Successful cases | Deterministic cases | Run |
| --- | --- | ---: | ---: | --- |
| akas (webgpu) | AMD Radeon RX 6700 XT | 253/258 | 255/258 | [20261117071744218931-50693886](https://github.com/kreasof-ai/akas/blob/b6b7d9aa78cf104b0894d6fcb1a620fdabde104c/benchmarks/runs/20261117071744218931-50693886/run.json) |
| llama (vulkan) | AMD Radeon RX 6700 XT (12272 MiB, 11474 MiB free) | 258/258 | 258/258 | [20261117064415142124-939fec36](https://github.com/kreasof-ai/akas/blob/ca7e489878e4442ed3329ac0352286ab471d1de9/benchmarks/runs/20261117064415142124-939fec36/run.json) |
