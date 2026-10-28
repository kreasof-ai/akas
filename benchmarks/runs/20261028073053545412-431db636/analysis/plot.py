"""Plot the matched phase medians retained in PARITY.json."""
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(sys.argv[1])
data=json.loads((root/'PARITY.json').read_text())
rows=data['rows']
plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none','font.size':10})
fig,axes=plt.subplots(1,2,figsize=(12,4.1),layout='constrained')
for ax,phase,title in zip(axes,['prefill','decode'],['Prefill','Decode']):
    for runtime,label,color,style in [('llama','llama.cpp Q4_0','#64748b','--'),('akas','AKAS Q4_0','#6750d5','-')]:
        ax.plot([r['prefix_tokens'] for r in rows],[r[phase][runtime] for r in rows],style,
                color=color,marker='o',markersize=4,linewidth=2,label=label)
    ax.set_xscale('log',base=2)
    ax.set_xticks([r['prefix_tokens'] for r in rows],['32','512','2k','8k','16k','33k','66k','131k'])
    ax.set_xlabel('Prefix tokens')
    ax.set_ylabel('Tokens / second')
    ax.set_title(title,loc='left',fontweight='bold')
    ax.grid(axis='y',alpha=.18)
    ax.spines[['top','right']].set_visible(False)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False)
fig.suptitle('LFM3-1.7B · QAD Q4_0 · NVIDIA L4 · one request, 128 output tokens',fontweight='bold')
fig.savefig(root/'throughput.svg')
fig.savefig(root/'throughput.png',dpi=150)
