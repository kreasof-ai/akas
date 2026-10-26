# Rejected development measurement

Do not use run `20261026145843633286-516a8e7e` for performance comparisons. Its timing protocol completed, but later validation found that its intermediate prefill graphs omitted eight required full-attention calls. The original sealed record remains immutable; this rejection supersedes its recorded timing-time pass.

The compiler appended a request-mapping input after a positional operation's output. A trailing output index then identified the mapping input instead of the actual attention result. State-prefill liveness dropped producers whose results remained necessary. The retained bad manifest shows nine full-attention calls in the complete phase and only one in the intermediate phase, where eight are required.

Commit `549c302` resolves lifted operation bindings by their original ABI names before adding mapping. A compiler regression checks a mapped read-only state operation, and the complete mixed-GGUF oracle now includes four layers with full attention in a middle layer. The new tests check both numerical results across chunk boundaries and retention of that layer in intermediate prefill.

The two-layer qualification `20261026145832920587-d5d98b7d` did not cover this middle-layer case and is superseded. The earlier native run `20261026140034040358-c39110df` retained all 36 attention calls in intermediate prefill and is not affected by this omission. The external llama.cpp run is also unaffected.

Only a new run of the corrected program may be used in the showcase. The rejected run and its source, samples and program remain archived for audit.
