"""Repeat affected profiles with cold-request host caches disabled."""
import json,subprocess,sys
from pathlib import Path
rows=[]
for provider in ('cuda','vulkan'):
 for fmt in ('f16','q4_0','q4_k_m'):
  args=['llama','qwen','--provider',provider,'--format',fmt,'--optimized','--seconds','900']
  print('RUN',args,flush=True)
  code=subprocess.call([sys.executable,'-m','tools.baselines',*args])
  rows.append(dict(args=args,returncode=code))
  Path('build/baselines/final-coverage.json').write_text(json.dumps(rows,indent=2)+'\n')
  print('DONE',code,args,flush=True)
