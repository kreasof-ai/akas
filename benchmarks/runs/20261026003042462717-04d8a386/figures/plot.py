"""Render a Spark X2.6 context comparison from its canonical portable report."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('report',type=Path);p.add_argument('out',type=Path);args=p.parse_args()
d=json.loads(args.report.read_text())
fig,axes=plt.subplots(1,2,figsize=(12,4.8))
for ax,metric,title in zip(axes,['prefill_tok_s','decode_tok_s'],['Prefill','Decode']):
 for profile,color,marker in zip(d['profiles'],['#2563eb','#ea580c'],['o','s']):
  rows=sorted(profile['measurements'],key=lambda x:x['prefix'])
  label='AKAS' if profile['kind']=='akas' else 'llama.cpp'
  ax.plot([x['prefix'] for x in rows],[x[metric] for x in rows],label=label,color=color,marker=marker,linewidth=2)
 ax.set_title(title,loc='left',fontweight='bold')
 ax.set_xscale('log',base=2);ax.set_ylim(bottom=0)
 ax.set_xticks([32,512,2048,8192,32768,130944],['32','512','2k','8k','32k','130,944'])
 ax.set_xlabel('Prefix tokens');ax.set_ylabel('Tokens / second')
 ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False)
axes[0].legend(frameon=False)
fig.suptitle('Spark X2.6 4B · Q4_K_M · NVIDIA L4',fontweight='bold',x=.06,ha='left')
fig.text(.06,.025,'FP16 KV · 1 request × 128 total outputs · median of 3 samples after 1 warmup per prefix\nIdentical checkpoint and synthetic input IDs. Native AKAS and llama-server phase timing boundaries differ.',fontsize=8,color='#475569')
fig.subplots_adjust(left=.06,right=.98,top=.82,bottom=.23,wspace=.24)
args.out.parent.mkdir(parents=True,exist_ok=True)
fig.savefig(args.out.with_suffix('.svg'),facecolor='white')
fig.savefig(args.out.with_suffix('.png'),dpi=160,facecolor='white')
