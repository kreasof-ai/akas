"""Render matched throughput with observed sample ranges and the requested prefill target."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('report',type=Path);p.add_argument('out',type=Path);args=p.parse_args()
d=json.loads(args.report.read_text());fig,axes=plt.subplots(1,2,figsize=(12,5))
for ax,metric,sample_key,title in zip(axes,['prefill_tok_s','decode_tok_s'],['prefill_tokens_per_second','decode_tokens_per_second'],['Prefill','Decode']):
 for profile,color,marker in zip(d['profiles'],['#2563eb','#ea580c'],['o','s']):
  rows=sorted(profile['measurements'],key=lambda x:x['prefix']);x=[r['prefix'] for r in rows];y=[r[metric] for r in rows]
  low=[r[metric]-min(s[sample_key] for s in r['samples']) for r in rows];high=[max(s[sample_key] for s in r['samples'])-r[metric] for r in rows]
  ax.errorbar(x,y,yerr=[low,high],label='AKAS' if profile['kind']=='akas' else 'llama.cpp',color=color,marker=marker,linewidth=2,capsize=3)
  if metric=='prefill_tok_s' and profile['kind']=='baseline':ax.plot(x,[2*v for v in y],label='2× llama.cpp target',color='#64748b',linestyle='--',linewidth=1.3)
 ax.set_title(title,loc='left',fontweight='bold');ax.set_xscale('log',base=2);ax.set_ylim(bottom=0)
 ax.set_xticks([32,512,2048,8192,32768,130944],['32','512','2k','8k','32k','130,944']);ax.set_xlabel('Prefix tokens');ax.set_ylabel('Tokens / second');ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False)
axes[0].legend(frameon=False,fontsize=8)
fig.suptitle('Spark X2.6 4B · Q4_K_M · NVIDIA L4',fontweight='bold',x=.06,ha='left')
fig.text(.06,.025,'FP16 KV · 1 request × 128 total outputs · median and observed range of 3 samples after 1 warmup\nIdentical checkpoint and synthetic input IDs. Native AKAS and llama-server phase timing boundaries differ.',fontsize=8,color='#475569')
fig.subplots_adjust(left=.06,right=.98,top=.84,bottom=.23,wspace=.24);args.out.parent.mkdir(parents=True,exist_ok=True)
for ext in ('.svg','.png'):fig.savefig(args.out.with_suffix(ext),dpi=160,facecolor='white')
