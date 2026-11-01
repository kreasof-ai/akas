# Bounded CUDA shapes: first implementation evidence

This pilot implements part of the [dynamic-shape plan](../../../plans/dynamic-shapes-and-batch-invariance.md).
It does not establish dynamic AKAS serving, batch invariance or an advantage
over vLLM/SGLang.

## Mechanisms

- Format-4 CUDA artifacts carry bounded shape contracts. The consumer rejects
  inadmissible calls before allocation/launch; the producer feeds the same facts
  into TIR assumptions. Old artifact versions keep their existing meaning.
- Canonical matmul accepts symbolic M with static tile sizes and N/K. CUDA
  prelowering follows the existing matmul path.
- CPU/CUDA buffers expose contiguous capacity prefixes. Public executable
  bindings preserve fixed descriptors for uncaptured submission and retain
  ancestor lifetime checks, including DLPack and CUDA graph resources.
- Multi-shape trials share each candidate build, qualify every sampled shape
  independently and release candidate resources before the next build.
- Region selection reuses the measured bank, with a region-count budget and
  per-point slowdown cap. Version-2 module exports verify complete bounded
  coverage, deterministic ordered overlaps and artifact guards. Several regions
  can share an image. Consumer selection imports no compiler.

## First measurements

NVIDIA L40S (`sm_89`), FP16 canonical matmul, N=1536, K=2048, M in 1..256.
Four dynamic candidates differ in block M (16/32/64/128). Each is checked
against NumPy at M=1,3,7,31,33,64,129,256. CUDA-event measurements use three
warmups, seven samples and ten repeats with prebound uncaptured calls.

The original exact controls use each point's winning dynamic schedule. They
are not independently tuned static controls and do not yet cover the compressed
portfolio's selected schedule at every point.

| M | Winning dynamic µs | Matched exact µs | Dynamic / exact |
| ---: | ---: | ---: | ---: |
| 1 | 11.568 | 21.504 | 0.538 |
| 3 | 11.571 | 11.264 | 1.027 |
| 7 | 11.776 | 11.878 | 0.991 |
| 31 | 11.283 | 11.162 | 1.011 |
| 33 | 11.571 | 11.274 | 1.026 |
| 64 | 12.285 | 11.149 | 1.102 |
| 129 | 15.872 | 15.770 | 1.006 |
| 256 | 20.890 | 23.245 | 0.899 |

These isolated samples show approximate parity at most points. They do not
support a general speedup claim; M=64 exceeds a 10% regression budget, and the
large M=1 difference needs repeat investigation. Producer wall time was 41.09 s
for four dynamic and eight exact builds, references and measurements combined.
Build-stage attribution and an independently tuned static baseline remain open.

Region selection compresses the bank into four intervals (1..19, 20..48,
49..192, 193..256) using three images. Its measured weighted latency is 1.0034x
the pointwise best dynamic candidates; its 1.10x cap applies to those dynamic
measurements, not the exact controls. A fresh consumer selected those packaged
images at previously unmeasured M=2,15,32,65,128,255 and passed independent
NumPy checks, including untouched unused output capacity. No performance
guarantee is inferred for unmeasured integers.

Raw [measurements and build records](../evidence/l40s-matmul-pilot-results.json)
and [fresh-consumer checks](../evidence/l40s-matmul-pilot-consumer-qualification.json)
retain the selected map and image hashes. The study script now compares exact
controls against the compressed portfolio's actual selected schedules. A fresh
serialized run is recorded below.

## Selected-portfolio rerun

The [second record](../evidence/l40s-matmul-portfolio-results.json) uses the same
shape points and candidate budget, with the exact control at each point using
the schedule actually chosen for that point's region. Three intervals use three
images; producer wall time is 34.78 s. Dynamic/exact latency ratios are 0.982,
0.991, 1.019, 1.030, 1.036, 1.056, 1.012 and 1.017 in ascending M order.
Every sampled point meets a 10% matched-schedule regression budget in this run.
This is a kernel pilot, not an inference or independently tuned-static
comparison. A separate compiler test proves guarded bounds eliminate an
impossible branch through TileLang's existing scoped-assumption pass.

CUDA qualification also exercises two independent runtime axes in one image,
including divisibility and relational guards, and a separate explicit live
extent over fixed-capacity buffers. Both preserve unused output capacity and
reject inadmissible bindings. This extends correctness evidence; it supplies no
multi-axis performance guarantee.

## Remaining acceptance work

Broader operation and multi-axis performance qualification, compiler simplification evidence,
independently tuned static controls, bounded replay/capture policy,
and workload-weighted operation coverage remain open. AKAS still needs symbolic
graph/call contracts, live-row execution and packed-token operations before the
plan's dynamic decode and packed prefill milestones can be called complete.
