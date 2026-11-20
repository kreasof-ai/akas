# Transcription comparison

[Overview](README.md) · [Comparison](COMPARISON.md) · [AKAS](AKAS.md) · [Baselines](BASELINES.md) · [Measurements](measurements.csv)

Warm wall time covers the request through receipt of the complete transcript over local process pipes. The timer includes WAV loading, feature extraction and greedy decoding. Model loading and context reset are outside the timer. RTF is wall seconds per second of input audio. WER and CER use case-folded NFKC text with punctuation removed. The clips are recorded speech from LibriSpeech dev-clean, validation subset, reader 1272. Both profiles are unqualified for independent full-model accuracy.

| Case | Duration band | Audio s | AKAS wall ms | llama.cpp wall ms | llama.cpp / AKAS | Exact transcript match |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1272-135031-0009 | short | 1.9 | 45.2 | 56.6 | 1.252 | yes |
| 1272-128104-0014 | short | 2.2 | 59.1 | 66.2 | 1.119 | yes |
| 1272-135031-0004 | short | 4.2 | 84.6 | 84.5 | 0.998 | yes |
| 1272-128104-0005 | medium | 9.0 | 158.6 | 131.6 | 0.830 | yes |
| 1272-135031-0000 | medium | 10.9 | 193.5 | 171.1 | 0.884 | yes |
| 1272-128104-0002 | medium | 12.5 | 202.9 | 177.3 | 0.874 | yes |
| 1272-128104-0011 | long | 15.1 | 250.3 | 201.0 | 0.803 | yes |
| 1272-128104-0009 | long | 18.3 | 294.4 | 243.5 | 0.827 | yes |
| 1272-128104-0004 | long | 29.4 | 490.5 | 377.5 | 0.769 | yes |

Band medians use clips completed deterministically by both runtimes. Ratios are geometric means of the paired per-clip ratios.

| Duration band | Completed clips | Audio s | AKAS wall ms | llama.cpp wall ms | llama.cpp / AKAS |
| --- | ---: | --- | ---: | ---: | ---: |
| short | 3/3 | 1.91–4.22 | 59.1 | 66.2 | 1.1 |
| medium | 3/3 | 9.01–12.48 | 193.5 | 171.1 | 0.9 |
| long | 3/3 | 15.12–29.40 | 294.4 | 243.5 | 0.8 |

Geometric mean wall-time ratio across 9 matched clips is 0.917×.
