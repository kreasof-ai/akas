"""One qualified expert projection for a cold-cache hardware-counter profile."""
import argparse,hashlib,json,zipfile
from pathlib import Path
import numpy as np
import tensor
from tensor.artifacts.format import read_artifact
from akas.artifacts.read import read_program,artifact_path
from akas.kernels.contracts import sample_arguments
from akas.quantization.weights.ggml.validation import sample_weight
from akas.ops.moe.validation import tile_tables
p=argparse.ArgumentParser();p.add_argument('--export-dir');p.add_argument('--rows',type=int,required=True);p.add_argument('--candidate',choices=('baseline','packed'),default='baseline');args=p.parse_args();rows=args.rows
program=Path('build/lfm-moe/codec-model/cuda/program');top=read_program(program);manifest=json.loads((program/next(r['path'] for r in top['phases'] if r['phase']=='prefill' and r['chunk']==rows)).read_text());record=next(k for k in manifest['kernels'].values() if k['kind']=='quantized_experts' and k['parameters']['k']==2048);baseline=artifact_path(program,record);signature,_=read_artifact(baseline);arrays=sample_arguments(signature['arguments']);sample_weight(arrays['w'],2,decode=False)
ids=np.argsort(np.random.default_rng(17).random((rows,64)),axis=1)[:,:4];counts=np.zeros(64,dtype='int32');routes=np.full((64,rows),-1,dtype='int32')
for expert in range(64):
 owners,ranks=np.where(ids[:-1]==expert);counts[expert]=len(owners);routes[expert,:len(owners)]=owners*4+ranks
arrays['counts'][:]=counts;arrays['routes'][:]=routes;tiles,local=tile_tables(counts,arrays['tile_experts'].shape[1]);arrays['tile_experts'][:]=tiles;arrays['tile_rows'][:]=local
path=baseline
if args.candidate=='packed':
 assert rows==4096;offsets=np.concatenate((np.zeros(1,dtype='int32'),np.cumsum(counts,dtype='int32')));x=np.zeros((rows*4,2048),dtype='float16')
 for expert in range(64):x[offsets[expert]:offsets[expert+1]]=arrays['x'][routes[expert,:counts[expert]]//4].astype('float16')
 arrays['x']=x;arrays['offsets']=offsets;path=Path('build/lfm-moe/packed-q4-expert/2048-128-128-128-256-1.tbin')
if args.export_dir:
 directory=Path(args.export_dir);directory.mkdir(parents=True,exist_ok=True);metadata,_=read_artifact(path)
 assert all(a['kind']=='buffer' for a in metadata['abi'])
 for a in metadata['arguments']:(directory/(a['name']+'.bin')).write_bytes(arrays[a['name']].tobytes())
 with zipfile.ZipFile(path) as z:(directory/'kernel.cubin').write_bytes(z.read('kernel.cubin'))
 spec=dict(entrypoint=metadata['entrypoint'],launch=metadata['launch'],abi=metadata['abi'],arguments=[a['name'] for a in metadata['arguments']],sha256=hashlib.sha256(path.read_bytes()).hexdigest(),counts=counts.tolist(),rows=rows,candidate=args.candidate)
 (directory/'target.json').write_text(json.dumps(spec,indent=2)+'\n');print('EXPORTED',rows,args.candidate,flush=True);raise SystemExit(0)
with tensor.Device(provider='cuda') as device:
 buffers={name:device.from_numpy(value) for name,value in arrays.items()}
 with device.load(path) as kernel:
  print('TARGET',json.dumps(dict(rows=rows,candidate=args.candidate,selected_experts=int((counts>0).sum()),counts=counts.tolist(),launch=kernel.manifest['launch'])),flush=True)
  kernel.launch(*[buffers[a['name']] for a in kernel.manifest['arguments']]);device.synchronize()
