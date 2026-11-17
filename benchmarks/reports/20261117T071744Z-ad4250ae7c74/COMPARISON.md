# Speech benchmark comparison

[Overview](README.md) · [Comparison](COMPARISON.md) · [AKAS](AKAS.md) · [Baselines](BASELINES.md) · [Measurements](measurements.csv)

Timings cover one warm localhost request through receipt of complete PCM16. Wall time and real-time factor (RTF, wall time divided by audio duration) are per-utterance medians. Comparison ratios use geometric means of matched successful utterances. Pass counts cover request completion, format and non-silent audio. AKAS also reports end of speech. WAV byte determinism is assessed within each runtime. All runtimes used one fixed voice for all languages. Intelligibility, pronunciation and MOS were outside the measurement scope.

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

## Relative to llama.cpp CPU

All three runtimes must complete a prompt for it to enter the ratios. Wall speedup is CPU wall time divided by the corresponding GPU wall time.

| Language | Cases | CPU passed | Matched on all three | AKAS wall speedup | llama.cpp Vulkan wall speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| English | 6 | 6 | 6 | 1.841× | 1.213× |
| French | 6 | 6 | 6 | 1.608× | 1.248× |
| Spanish | 6 | 6 | 6 | 1.987× | 1.622× |
| German | 6 | 6 | 6 | 1.731× | 1.139× |
| Italian | 6 | 6 | 6 | 1.596× | 1.021× |
| Portuguese | 6 | 6 | 6 | 1.806× | 1.110× |
| Dutch | 6 | 6 | 6 | 1.735× | 1.149× |
| Greek | 6 | 6 | 6 | 2.314× | 1.735× |
| Swedish | 6 | 6 | 6 | 1.948× | 1.209× |
| Danish | 6 | 6 | 6 | 1.846× | 1.233× |
| Norwegian Bokmål | 6 | 6 | 6 | 1.931× | 1.517× |
| Finnish | 6 | 6 | 6 | 1.586× | 1.346× |
| Polish | 6 | 6 | 6 | 1.993× | 1.370× |
| Czech | 6 | 6 | 6 | 1.659× | 1.250× |
| Slovak | 6 | 6 | 6 | 1.620× | 1.072× |
| Hungarian | 6 | 6 | 6 | 2.343× | 1.265× |
| Romanian | 6 | 6 | 6 | 1.938× | 1.074× |
| Bulgarian | 6 | 6 | 6 | 1.727× | 1.113× |
| Russian | 6 | 6 | 6 | 1.746× | 1.974× |
| Ukrainian | 6 | 6 | 6 | 1.900× | 1.615× |
| Turkish | 6 | 6 | 6 | 1.785× | 1.122× |
| Arabic | 6 | 6 | 6 | 1.289× | 1.289× |
| Hebrew | 6 | 6 | 6 | 1.594× | 1.035× |
| Georgian | 6 | 6 | 6 | 2.144× | 1.483× |
| Mandarin Chinese | 6 | 6 | 6 | 1.321× | 1.246× |
| Japanese | 6 | 6 | 4 | 1.334× | 1.109× |
| Korean | 6 | 6 | 6 | 1.078× | 0.999× |
| Vietnamese | 6 | 6 | 6 | 1.800× | 1.511× |
| Indonesian | 6 | 6 | 6 | 2.060× | 1.113× |
| Malay | 6 | 6 | 6 | 1.448× | 1.015× |
| Thai | 6 | 6 | 6 | 2.037× | 0.954× |
| Tagalog | 6 | 6 | 6 | 1.741× | 1.125× |
| Hindi | 6 | 6 | 6 | 1.508× | 0.895× |
| Bengali | 6 | 6 | 6 | 1.697× | 0.998× |
| Tamil | 6 | 6 | 6 | 1.370× | 1.013× |
| Telugu | 6 | 6 | 6 | 1.985× | 1.073× |
| Marathi | 6 | 6 | 6 | 1.311× | 0.946× |
| Gujarati | 6 | 6 | 6 | 1.059× | 1.218× |
| Kannada | 6 | 6 | 6 | 1.390× | 1.484× |
| Malayalam | 6 | 6 | 6 | 1.090× | 0.873× |
| Punjabi | 6 | 6 | 3 | 2.023× | 2.398× |
| Odia | 6 | 6 | 6 | 1.833× | 1.497× |
| Urdu | 6 | 6 | 6 | 1.699× | 1.730× |
| All languages | 258 | 258 | 253 | 1.678× | 1.226× |
