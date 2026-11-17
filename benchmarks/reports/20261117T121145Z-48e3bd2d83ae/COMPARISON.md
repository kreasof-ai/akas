# Speech benchmark comparison

[Overview](README.md) · [Comparison](COMPARISON.md) · [AKAS](AKAS.md) · [Baselines](BASELINES.md) · [Measurements](measurements.csv)

Timings cover one warm localhost request through receipt of complete PCM16. Wall time and real-time factor (RTF, wall time divided by audio duration) are per-utterance medians. Comparison ratios use geometric means of matched successful utterances. Pass counts cover request completion, format and non-silent audio. AKAS also reports end of speech. WAV byte determinism is assessed within each runtime. All runtimes used one fixed voice for all languages. Intelligibility, pronunciation and MOS were outside the measurement scope.

| Language | Cases | AKAS passed | Baseline passed | Paired | Wall wins | Wall speedup | Audio throughput speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| English (`en`) | 6 | 6 | 6 | 6 | 6 | 1.760× | 1.811× |
| French (`fr`) | 6 | 6 | 6 | 6 | 6 | 1.684× | 1.783× |
| Spanish (`es`) | 6 | 6 | 6 | 6 | 5 | 1.500× | 2.075× |
| German (`de`) | 6 | 6 | 6 | 6 | 6 | 1.884× | 1.846× |
| Italian (`it`) | 6 | 6 | 6 | 6 | 6 | 1.839× | 1.858× |
| Portuguese (`pt`) | 6 | 6 | 6 | 6 | 6 | 1.709× | 1.654× |
| Dutch (`nl`) | 6 | 6 | 6 | 6 | 6 | 1.652× | 1.618× |
| Greek (`el`) | 6 | 6 | 6 | 6 | 5 | 1.250× | 1.625× |
| Swedish (`sv`) | 6 | 6 | 6 | 6 | 6 | 1.842× | 1.805× |
| Danish (`da`) | 6 | 6 | 6 | 6 | 6 | 1.651× | 1.661× |
| Norwegian Bokmål (`nb`) | 6 | 6 | 6 | 6 | 6 | 1.544× | 1.558× |
| Finnish (`fi`) | 6 | 6 | 6 | 6 | 5 | 1.479× | 2.006× |
| Polish (`pl`) | 6 | 6 | 6 | 6 | 5 | 1.917× | 1.699× |
| Czech (`cs`) | 6 | 6 | 6 | 6 | 5 | 1.509× | 1.674× |
| Slovak (`sk`) | 6 | 6 | 6 | 6 | 6 | 1.959× | 1.474× |
| Hungarian (`hu`) | 6 | 6 | 6 | 6 | 5 | 1.993× | 1.538× |
| Romanian (`ro`) | 6 | 6 | 6 | 6 | 6 | 1.801× | 1.500× |
| Bulgarian (`bg`) | 6 | 6 | 6 | 6 | 6 | 1.755× | 1.823× |
| Russian (`ru`) | 6 | 6 | 6 | 6 | 2 | 0.933× | 2.012× |
| Ukrainian (`uk`) | 6 | 6 | 6 | 6 | 5 | 1.429× | 1.780× |
| Turkish (`tr`) | 6 | 6 | 6 | 6 | 6 | 1.937× | 1.869× |
| Arabic (`ar`) | 6 | 6 | 6 | 6 | 4 | 1.299× | 1.862× |
| Hebrew (`he`) | 6 | 6 | 6 | 6 | 6 | 1.900× | 1.440× |
| Georgian (`ka`) | 6 | 5 | 6 | 5 | 3 | 1.163× | 1.816× |
| Mandarin Chinese (`zh`) | 6 | 6 | 6 | 6 | 5 | 1.332× | 1.773× |
| Japanese (`ja`) | 6 | 6 | 6 | 6 | 3 | 1.028× | 2.060× |
| Korean (`ko`) | 6 | 6 | 6 | 6 | 3 | 1.077× | 1.895× |
| Vietnamese (`vi`) | 6 | 6 | 6 | 6 | 5 | 1.400× | 1.903× |
| Indonesian (`id`) | 6 | 6 | 6 | 6 | 6 | 1.836× | 1.720× |
| Malay (`ms`) | 6 | 6 | 6 | 6 | 6 | 1.861× | 1.820× |
| Thai (`th`) | 6 | 6 | 6 | 6 | 6 | 2.003× | 1.679× |
| Tagalog (`tl`) | 6 | 6 | 6 | 6 | 6 | 1.695× | 1.643× |
| Hindi (`hi`) | 6 | 6 | 6 | 6 | 6 | 1.854× | 1.386× |
| Bengali (`bn`) | 6 | 5 | 6 | 5 | 5 | 1.898× | 1.593× |
| Tamil (`ta`) | 6 | 6 | 6 | 6 | 5 | 1.421× | 1.554× |
| Telugu (`te`) | 6 | 6 | 6 | 6 | 6 | 2.011× | 1.289× |
| Marathi (`mr`) | 6 | 6 | 6 | 6 | 6 | 1.663× | 1.797× |
| Gujarati (`gu`) | 6 | 6 | 6 | 6 | 5 | 1.206× | 1.734× |
| Kannada (`kn`) | 6 | 6 | 6 | 6 | 4 | 0.967× | 1.581× |
| Malayalam (`ml`) | 6 | 6 | 6 | 6 | 4 | 1.266× | 1.909× |
| Punjabi (`pa`) | 6 | 3 | 6 | 3 | 2 | 1.314× | 1.282× |
| Odia (`or`) | 6 | 6 | 6 | 6 | 6 | 1.280× | 1.500× |
| Urdu (`ur`) | 6 | 6 | 6 | 6 | 2 | 0.907× | 1.950× |
| **All languages** | 258 | 253 | 258 | 253 | 219 | 1.534× | 1.712× |

## Relative to llama.cpp CPU

All three runtimes must complete a prompt for it to enter the ratios. Wall speedup is CPU wall time divided by the corresponding GPU wall time.

| Language | Cases | CPU passed | Matched on all three | AKAS wall speedup | llama.cpp Vulkan wall speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| English | 6 | 6 | 6 | 2.135× | 1.213× |
| French | 6 | 6 | 6 | 2.102× | 1.248× |
| Spanish | 6 | 6 | 6 | 2.433× | 1.622× |
| German | 6 | 6 | 6 | 2.146× | 1.139× |
| Italian | 6 | 6 | 6 | 1.878× | 1.021× |
| Portuguese | 6 | 6 | 6 | 1.897× | 1.110× |
| Dutch | 6 | 6 | 6 | 1.899× | 1.149× |
| Greek | 6 | 6 | 6 | 2.169× | 1.735× |
| Swedish | 6 | 6 | 6 | 2.227× | 1.209× |
| Danish | 6 | 6 | 6 | 2.035× | 1.233× |
| Norwegian Bokmål | 6 | 6 | 6 | 2.343× | 1.517× |
| Finnish | 6 | 6 | 6 | 1.991× | 1.346× |
| Polish | 6 | 6 | 6 | 2.628× | 1.370× |
| Czech | 6 | 6 | 6 | 1.886× | 1.250× |
| Slovak | 6 | 6 | 6 | 2.100× | 1.072× |
| Hungarian | 6 | 6 | 6 | 2.522× | 1.265× |
| Romanian | 6 | 6 | 6 | 1.935× | 1.074× |
| Bulgarian | 6 | 6 | 6 | 1.953× | 1.113× |
| Russian | 6 | 6 | 6 | 1.842× | 1.974× |
| Ukrainian | 6 | 6 | 6 | 2.308× | 1.615× |
| Turkish | 6 | 6 | 6 | 2.173× | 1.122× |
| Arabic | 6 | 6 | 6 | 1.674× | 1.289× |
| Hebrew | 6 | 6 | 6 | 1.967× | 1.035× |
| Georgian | 6 | 6 | 5 | 1.760× | 1.514× |
| Mandarin Chinese | 6 | 6 | 6 | 1.660× | 1.246× |
| Japanese | 6 | 6 | 6 | 0.925× | 0.900× |
| Korean | 6 | 6 | 6 | 1.077× | 0.999× |
| Vietnamese | 6 | 6 | 6 | 2.115× | 1.511× |
| Indonesian | 6 | 6 | 6 | 2.042× | 1.113× |
| Malay | 6 | 6 | 6 | 1.889× | 1.015× |
| Thai | 6 | 6 | 6 | 1.910× | 0.954× |
| Tagalog | 6 | 6 | 6 | 1.907× | 1.125× |
| Hindi | 6 | 6 | 6 | 1.659× | 0.895× |
| Bengali | 6 | 6 | 5 | 2.186× | 1.152× |
| Tamil | 6 | 6 | 6 | 1.441× | 1.013× |
| Telugu | 6 | 6 | 6 | 2.158× | 1.073× |
| Marathi | 6 | 6 | 6 | 1.573× | 0.946× |
| Gujarati | 6 | 6 | 6 | 1.469× | 1.218× |
| Kannada | 6 | 6 | 6 | 1.435× | 1.484× |
| Malayalam | 6 | 6 | 6 | 1.105× | 0.873× |
| Punjabi | 6 | 6 | 3 | 3.152× | 2.398× |
| Odia | 6 | 6 | 6 | 1.915× | 1.497× |
| Urdu | 6 | 6 | 6 | 1.570× | 1.730× |
| All languages | 258 | 258 | 253 | 1.876× | 1.223× |
