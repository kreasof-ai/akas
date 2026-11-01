from pathlib import Path
import json,time
import numpy as np
import tensor
from akas.models.lfm2 import LFM2MoE
root=Path('build/lfm-serving');baseline=json.loads((root/'sglang-text-probes.json').read_text());rows=[];arrays={}
with tensor.Device() as device,LFM2MoE('build/checkpoints/lfm2.5-24b-a2b-fp8',root/'program-concurrent',device,checkpoint_options={'tokenizer':False}) as model:
 for index,row in enumerate(baseline['rows']):
  slot=model.admit()
  predicted=model.prefill([slot],[row['tokens']],chunk=2048)[0]
  logits=model.last_frame.feature('logits').to_numpy().reshape(-1)
  z=logits.astype('float64');z-=z.max();logp=z-np.log(np.exp(z).sum())
  target=row['response']['output_ids'][0]
  probe=dict(text=row['text'],bos=row['bos'],prediction=predicted,baseline_prediction=target,greedy_equal=predicted==target,baseline_token_probability=float(np.exp(logp[target])),top10=[int(i) for i in np.argsort(-logits)[:10]])
  print('PROBE',probe,flush=True);rows.append(probe);arrays[f'logits{index}']=logits.copy()
  model.release(slot)
 np.savez(root/'native-text-probe-logits.npz',**arrays)
 (root/'native-text-probes.json').write_text(json.dumps(dict(runtime='akas',diagnostic=True,model_numerical_qualification='open',rows=rows),indent=2)+'\n')
