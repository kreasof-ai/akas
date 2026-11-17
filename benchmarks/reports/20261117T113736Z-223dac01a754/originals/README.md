# Speech benchmark report

[Overview](README.md) · [Comparison](COMPARISON.md) · [AKAS](AKAS.md) · [Baselines](BASELINES.md) · [Measurements](measurements.csv)

Workload `speech-language-voices-v1` covers **10 languages**, **60 utterances** and 2 measured requests per utterance after 1 warmup.

Protocol `client-speech-pcm-v1`. UTC 2026-11-17T11:37:36.893998Z to 2026-11-17T11:41:31.764037Z.

Timings cover one warm localhost request through receipt of complete PCM16. Wall time and real-time factor (RTF, wall time divided by audio duration) are per-utterance medians. Comparison ratios use geometric means of matched successful utterances. Pass counts cover request completion, format and non-silent audio. AKAS also reports end of speech. WAV byte determinism is assessed within each runtime. Both runtimes used the same case-specific presets. Intelligibility, pronunciation and MOS were outside the measurement scope.

Language coverage follows the [model card](https://huggingface.co/Zyphra/ZONOS3-Mini#languages). Prompts are from [FLORES-200 devtest](https://github.com/facebookresearch/flores/tree/main/flores200), licensed [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Six aligned one-based rows, two per length tier by English character count. Rows use translated text as distributed and exclude digits across the selected languages.

| Language | Preset voice |
| --- | --- |
| English (`en`) | Savannah_en-American_F_1 |
| French (`fr`) | Gaspard_fr_M_1 |
| Spanish (`es`) | Rodrigo_es_M_1 |
| German (`de`) | Liesel_de_F_1 |
| Italian (`it`) | Ginevra_it_F_1 |
| Portuguese (`pt`) | Leonor_pt_F_1 |
| Mandarin Chinese (`zh`) | Haoran_zh-energetic_M_1 |
| Korean (`ko`) | Minjun_ko_M_1 |
| Hindi (`hi`) | Ananya_hi_F_1 |
| Telugu (`te`) | Chaitanya_te_M_1 |

| Runtime | Device | Successful cases | Deterministic cases | Run |
| --- | --- | ---: | ---: | --- |
| akas (webgpu) | AMD Radeon RX 6700 XT | 60/60 | 60/60 | [20261117113736893998-27a26a5b](../../runs/20261117113736893998-27a26a5b/run.json) |
| llama (vulkan) | AMD Radeon RX 6700 XT (12272 MiB, 11474 MiB free) | 60/60 | 60/60 | [20261117075749615890-46300cb5](../../runs/20261117075749615890-46300cb5/run.json) |
