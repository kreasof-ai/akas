# AKAS transcription measurements

[Overview](README.md) · [Comparison](COMPARISON.md) · [AKAS](AKAS.md) · [Baselines](BASELINES.md) · [Measurements](measurements.csv)

Warm wall time covers the request through receipt of the complete transcript over local process pipes. The timer includes WAV loading, feature extraction and greedy decoding. Model loading and context reset are outside the timer. RTF is wall seconds per second of input audio. WER and CER use case-folded NFKC text with punctuation removed. The clips are recorded speech from LibriSpeech dev-clean, validation subset, reader 1272. Both profiles are unqualified for independent full-model accuracy.

Run [20261120084341194215-d25a4c19](https://github.com/kreasof-ai/akas/blob/29bc2bcf9e670cdd9ddb3f988bdf7626ebbd6cc8/benchmarks/runs/20261120084341194215-d25a4c19/run.json). Report SHA-256 `1abfad4445bc250a772a19744b1b4d2be0affbe45e5c2439a49a6c7f540ef324`.
Model Qwen3 ASR (Rumble-2-Flash). Format `q8_0`. Context 512 tokens.
Decoder SHA-256 `600821d04cd1a7caa42f4622f6ac42b84e6db00f73196ed73493d06c221886bc`. Projector SHA-256 `97116d3656cbe4912ac9aee43eb4a887832121e2a2c07c5ee61617052bdd30f8`.
Each clip has 1 warmup and 3 measured requests.
Compiled program SHA-256 `edc82d076e55d35ffea4ed321ce4f321f02cd56b41e633d54c3a0c9a167bde3e`.

| Case | Status | Wall ms | Audio s | RTF | Error rate | Transcript |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1272-135031-0009 | ok | 45.2 | 1.9 | 0.024 | WER 0.0% | He doesn't work at all. |
| 1272-128104-0014 | ok | 59.1 | 2.2 | 0.026 | WER 0.0% | By Harry Quilter, M. A. |
| 1272-135031-0004 | ok | 84.6 | 4.2 | 0.020 | WER 0.0% | The king has fled in disgrace, and your friends are asking for you. |
| 1272-128104-0005 | ok | 158.6 | 9.0 | 0.018 | WER 0.0% | It is obviously unnecessary for us to point out how luminous these criticisms are, how delicate in expression. |
| 1272-135031-0000 | ok | 193.5 | 10.9 | 0.018 | WER 3.8% | Because you are sleeping instead of conquering, the lovely rose princess has become a fiddle without a bow, while poor Shaggy sits there, a cooing dove. |
| 1272-128104-0002 | ok | 202.9 | 12.5 | 0.016 | WER 0.0% | He tells us that at this festive season of the year, with Christmas and roast beef looming before us, similes drawn from eating and its results occur most readily to the mind. |
| 1272-128104-0011 | ok | 250.3 | 15.1 | 0.017 | WER 8.8% | In fact, he is quite severe on Mr. Ruskin for not recognizing that a picture should denote the frailty of man, and remarks with pleasing courtesy and felicitous grace that many phases of feeling are |
| 1272-128104-0009 | ok | 294.4 | 18.3 | 0.016 | WER 7.0% | He laments most bitterly the divorce that has been made between decorative art and what we usually call pictures, makes a customary appeal to the last judgment and reminds us that in the great days of art, Michelangelo was the furnishing upholsterer. |
| 1272-128104-0004 | ok | 490.5 | 29.4 | 0.017 | WER 9.9% | Lynell's pictures are a sort of upguard's and Adam paintings, and Mason's exquisite idylls are as national as a jingo poem. Mister Birkenfoster's landscapes smile at one much in the same way that Mister Carker used to flash his teeth, and Mister John Collier gives his sitter a cheerful slap on the back before he says, like a shampooer in a Turkish bath, "Next man." |
