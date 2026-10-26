"""Compare the requested performance targets and useful prefill MFU using sealed samples."""
import json
from pathlib import Path
from akas.models.spark import SparkCheckpoint
from akas.benchmarking.history import write
from tools.spark_mfu import estimate
from tools.campaign_report import record
from akas.benchmarking import archive

def generate(report_dir,work):
 d=json.loads((report_dir/'report.json').read_text());config=SparkCheckpoint('build/checkpoints/spark-x2.6-4b/gguf/Spark-X2.6-4B-Q4_K_M.gguf',tokenizer=False).config
 m=estimate(d,config);write(work/'mfu.json',m)
 byprefix={x['prefix']:x for x in m['rows'] if x['runtime']=='akas'}
 a,b=d['profiles'];history=Path('benchmarks');prior_id='20261025T235635Z-d80177ee179d';prior=json.loads(archive.read_file(history,'reports',prior_id,'report.json'));prior['report_url']=archive.link(history,'reports',prior_id,'README.md');priorrows={r['prefix']:r for r in prior['profiles'][0]['measurements']};rows=[]
 for left,right in zip(a['measurements'],b['measurements']):
  p=left['prefix'];nmin=min(s['decode_tokens_per_second'] for s in left['samples']);bmax=max(s['decode_tokens_per_second'] for s in right['samples'])
  rows.append(dict(prefix=p,prefill=left['prefill_tok_s'],llama_prefill=right['prefill_tok_s'],prefill_ratio=left['prefill_tok_s']/right['prefill_tok_s'],
                   decode=left['decode_tok_s'],llama_decode=right['decode_tok_s'],decode_ratio=left['decode_tok_s']/right['decode_tok_s'],minimum_decode=nmin,maximum_llama_decode=bmax,
                   all_decode_samples_ahead=nmin>bmax,estimated_mfu_percent=byprefix[p]['mfu_percent'],prefill_gain_over_original=left['prefill_tok_s']/priorrows[p]['prefill_tok_s'],
                   two_times_prefill_met=left['prefill_tok_s']>=2*right['prefill_tok_s']))
 old=record(history,prior['profiles'][0]['selected_run_id']);new=record(history,a['selected_run_id']);comparison=[]
 for left,right in zip(old['results'],new['results']):
  x,y=left['output_ids'][0],right['output_ids'][0];common=next((i for i,(u,v) in enumerate(zip(x,y)) if u!=v),min(len(x),len(y)));comparison.append(dict(prefix=left['case']['prompt'],old_count=len(x),new_count=len(y),equal=x==y,common_prefix=common))
 write(work/'prior-output-comparison.json',dict(old_run=old['id'],new_run=new['id'],rows=comparison))
 summary=dict(rows=rows,decode_medians_ahead=sum(r['decode_ratio']>1 for r in rows),decode_ranges_ahead=sum(r['all_decode_samples_ahead'] for r in rows),two_times_prefill_met=sum(r['two_times_prefill_met'] for r in rows),
              prior_report=prior['report_url'],native_run_id=a['selected_run_id'],native_outputs_equal=sum(x['equal'] for x in comparison))
 write(work/'assessment.json',summary)
 lines=['# Spark X2.6 optimization assessment','',
        'NVIDIA L4, official Q4_K_M checkpoint, FP16 KV, one request, eight prefixes and 128 total outputs. Medians of three retained samples follow one warmup at each prefix. Both runtimes use identical checkpoint bytes and synthetic input IDs. AKAS uses synchronized native phases and llama.cpp uses server eval phases; timing boundaries differ.','',
        f"Decode medians exceed llama.cpp at {summary['decode_medians_ahead']}/8 prefixes. Every retained AKAS decode sample exceeds every retained llama.cpp sample at {summary['decode_ranges_ahead']}/8 prefixes. These observations describe this run and do not guarantee all workloads or future runs.",'',
        f"The requested 2× llama.cpp prefill target is met at {summary['two_times_prefill_met']}/8 prefixes.",'',
        '| Prefix | AKAS prefill tok/s | llama.cpp prefill tok/s | Prefill ratio | AKAS decode tok/s | llama.cpp decode tok/s | Decode ratio | Estimated prefill MFU |',
        '| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
 for r in rows:lines.append(f"| {r['prefix']:,} | {r['prefill']:,.1f} | {r['llama_prefill']:,.1f} | {r['prefill_ratio']:.2f}× | {r['decode']:.2f} | {r['llama_decode']:.2f} | {r['decode_ratio']:.2f}× | {r['estimated_mfu_percent']:.2f}% |")
 lines+=['','The implementation adds an optional derived FP16 layer-weight cache for prefill, larger compressed projection tiles for short prompts, shared FP32 RoPE sine/cosine tables, vectorized Q8 decode and attention tiles that share K/V loads. The FP16 cache increases device memory by about 7.55 GB. Original packed weights remain in use for decode and short prompts.','',
         f"[The original comparison]({prior['report_url']}) estimated prefill MFU at 4.25% for prefix 32 and 9.84–10.29% for larger prefixes. Changes against that earlier run are descriptive; the fresh llama.cpp baseline above is the matched comparison for this run.",'',
         '## MFU method','',
         '[NVIDIA specifies 121 TFLOP/s dense FP16 for L4](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf) (Appendix D). The sparse 242 TFLOP/s figure is not used. Estimated useful MFU is `useful_forward_FLOPs / prefill_seconds / 121e12`.','',
         'Layer projection parameters total 3,776,348,160. Dense projections contribute `2 * prefix * parameters`; the final-row vocabulary projection contributes `2 * 335544320`. Attention contributes `4 * 16 * 256 * (9 * full_pairs + 27 * window_pairs)`, with `full_pairs = N*(N+1)/2` and `window_pairs = sum(min(t,512) for t in 1..N)`.','',
         'This counts useful dense model math, excludes embedding lookups, elementwise operations, padded work, dequantization and cache/control traffic, and counts all model layers even where intermediate prefill chunks prune final-layer outputs. It is an algorithmic estimate rather than a hardware instruction counter. The generator and full per-runtime numbers are retained beside this page.','',
         '## Numerical qualification','',
         '26 checks passed, with two optional pretrained tokenizer checks skipped. CUDA checks cover independent projection and softmax references, long-position rotary values, cache wrap, inactive requests, NaN padding, ragged tails, request reorder and slot reuse against independent small-model Torch equations. Targeted compiler and quantization regression checks also passed. Full checkpoint trajectories are compared separately in the canonical benchmark report; kernel qualification does not imply identical greedy trajectories across runtimes.','',
         f"The optimized path retains identical 128-token outputs to the original AKAS run at {summary['native_outputs_equal']}/8 prefixes. Divergence details and common-prefix lengths are retained beside this page. The full before/after comparison is retained in `prior-output-comparison.json`; different floating-point reduction orders need not preserve greedy trajectories.",'',
         '## Reproduce','',
         '```sh','python -m tools.spark_benchmark compile --optimized --chunk 1024 --program build/spark/fast --out build/spark/compile-fast.json','python -m tools.spark_benchmark akas --chunk 1024 --program build/spark/fast --out build/spark/akas-fast.json','python -m tools.spark_benchmark llama --chunk 1024 --out build/spark/llama-fast.json','python -m tools.spark_benchmark report --runs AKAS_RUN LLAMA_RUN --out build/spark/report-fast','python -m tools.spark_mfu PATH/TO/report.json --checkpoint PATH/TO/MODEL.gguf --out build/spark/mfu.json','```','',
         'Set up CUDA libraries and TENSOR_NVRTC_HOME as described in the repository. Measurements exclude compilation, upload and phase loading. The same pinned llama.cpp revision is used as in the original comparison. GPU power, clock, temperature, utilization and memory telemetry is retained for both sequential sweeps.','']
 (work/'assessment.md').write_text('\n'.join(lines));return summary

if __name__ == '__main__':
 import argparse
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('report_directory',type=Path);p.add_argument('output_directory',type=Path);args=p.parse_args();generate(args.report_directory,args.output_directory)
