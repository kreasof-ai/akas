# Speech benchmark comparison

[Overview](README.md) · [Comparison](COMPARISON.md) · [AKAS](AKAS.md) · [Baselines](BASELINES.md) · [Measurements](measurements.csv)

Timings cover one warm localhost request through receipt of complete PCM16. Wall time and real-time factor (RTF, wall time divided by audio duration) are per-utterance medians. Comparison ratios use geometric means of matched successful utterances. Pass counts cover request completion, format and non-silent audio. AKAS also reports end of speech. WAV byte determinism is assessed within each runtime. Both runtimes used one fixed voice for all languages. Intelligibility, pronunciation and MOS were outside the measurement scope.

| Language | Cases | AKAS passed | Baseline passed | Paired | Wall wins | Wall speedup | Audio throughput speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| English (`en`) | 6 | 6 | 6 | 6 | 6 | 1.518× | 1.576× |
| French (`fr`) | 6 | 6 | 6 | 6 | 4 | 1.289× | 1.476× |
| Spanish (`es`) | 6 | 6 | 6 | 6 | 5 | 1.225× | 1.734× |
| German (`de`) | 6 | 6 | 6 | 6 | 6 | 1.519× | 1.519× |
| Italian (`it`) | 6 | 6 | 6 | 6 | 5 | 1.562× | 1.514× |
| Portuguese (`pt`) | 6 | 6 | 6 | 6 | 6 | 1.626× | 1.397× |
| Dutch (`nl`) | 6 | 6 | 6 | 6 | 6 | 1.510× | 1.468× |
| Greek (`el`) | 6 | 6 | 6 | 6 | 4 | 1.334× | 1.395× |
| Swedish (`sv`) | 6 | 6 | 6 | 6 | 6 | 1.611× | 1.540× |
| Danish (`da`) | 6 | 6 | 6 | 6 | 6 | 1.498× | 1.502× |
| Norwegian Bokmål (`nb`) | 6 | 6 | 6 | 6 | 5 | 1.273× | 1.415× |
| Finnish (`fi`) | 6 | 6 | 6 | 6 | 5 | 1.178× | 1.629× |
| Polish (`pl`) | 6 | 6 | 6 | 6 | 5 | 1.455× | 1.587× |
| Czech (`cs`) | 6 | 6 | 6 | 6 | 5 | 1.328× | 1.509× |
| Slovak (`sk`) | 6 | 6 | 6 | 6 | 5 | 1.511× | 1.227× |
| Hungarian (`hu`) | 6 | 6 | 6 | 6 | 5 | 1.852× | 1.432× |
| Romanian (`ro`) | 6 | 6 | 6 | 6 | 6 | 1.804× | 1.538× |
| Bulgarian (`bg`) | 6 | 6 | 6 | 6 | 6 | 1.552× | 1.543× |
| Russian (`ru`) | 6 | 6 | 6 | 6 | 2 | 0.884× | 1.614× |
| Ukrainian (`uk`) | 6 | 6 | 6 | 6 | 4 | 1.176× | 1.334× |
| Turkish (`tr`) | 6 | 6 | 6 | 6 | 6 | 1.591× | 1.584× |
| Arabic (`ar`) | 6 | 6 | 6 | 6 | 3 | 1.000× | 1.813× |
| Hebrew (`he`) | 6 | 6 | 6 | 6 | 5 | 1.540× | 1.239× |
| Georgian (`ka`) | 6 | 6 | 6 | 6 | 5 | 1.445× | 1.574× |
| Mandarin Chinese (`zh`) | 6 | 6 | 6 | 6 | 5 | 1.060× | 1.672× |
| Japanese (`ja`) | 6 | 4 | 6 | 4 | 3 | 1.203× | 1.547× |
| Korean (`ko`) | 6 | 6 | 6 | 6 | 3 | 1.078× | 1.582× |
| Vietnamese (`vi`) | 6 | 6 | 6 | 6 | 5 | 1.192× | 1.622× |
| Indonesian (`id`) | 6 | 6 | 6 | 6 | 6 | 1.852× | 1.527× |
| Malay (`ms`) | 6 | 6 | 6 | 6 | 6 | 1.426× | 1.386× |
| Thai (`th`) | 6 | 6 | 6 | 6 | 6 | 2.136× | 1.274× |
| Tagalog (`tl`) | 6 | 6 | 6 | 6 | 6 | 1.548× | 1.523× |
| Hindi (`hi`) | 6 | 6 | 6 | 6 | 6 | 1.686× | 1.269× |
| Bengali (`bn`) | 6 | 6 | 6 | 6 | 6 | 1.700× | 1.260× |
| Tamil (`ta`) | 6 | 6 | 6 | 6 | 4 | 1.352× | 1.378× |
| Telugu (`te`) | 6 | 6 | 6 | 6 | 6 | 1.850× | 1.159× |
| Marathi (`mr`) | 6 | 6 | 6 | 6 | 6 | 1.386× | 1.404× |
| Gujarati (`gu`) | 6 | 6 | 6 | 6 | 2 | 0.869× | 1.545× |
| Kannada (`kn`) | 6 | 6 | 6 | 6 | 3 | 0.937× | 1.419× |
| Malayalam (`ml`) | 6 | 6 | 6 | 6 | 4 | 1.248× | 1.548× |
| Punjabi (`pa`) | 6 | 3 | 6 | 3 | 1 | 0.844× | 1.574× |
| Odia (`or`) | 6 | 6 | 6 | 6 | 5 | 1.225× | 1.471× |
| Urdu (`ur`) | 6 | 6 | 6 | 6 | 3 | 0.982× | 1.555× |
| **All languages** | 258 | 253 | 258 | 253 | 207 | 1.369× | 1.477× |
