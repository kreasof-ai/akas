import gzip,hashlib,json,urllib.request
from pathlib import Path
from akas.models.lfm2.model import LFM2MoE
from akas.benchmarking.history import write
from akas.benchmarking.workload import prompt_digest
out=Path('build/lfm-moe/c4-final');rev='1588ec454efa1a09f29cd18ddd04fe05fc8653a2';file='en/c4-validation.00000-of-00008.json.gz';url=f'https://huggingface.co/datasets/allenai/c4/resolve/{rev}/{file}'
reader=LFM2MoE.checkpoint_type('build/checkpoints/lfm2.5-24b-a2b/q4_0.gguf');texts=[];sources=[];chars=0
with urllib.request.urlopen(url,timeout=60) as response,gzip.GzipFile(fileobj=response) as stream:
 for i,line in enumerate(stream):
  d=json.loads(line);texts.append(d['text']);chars+=len(d['text']);sources.append(dict(row=i,text_sha256=hashlib.sha256(d['text'].encode()).hexdigest()))
  if chars>=900000:break
text='\n\n'.join(texts);tokens=list(map(int,reader.tokenizer.encode(text)))
if len(tokens)<130944:raise ValueError(f'Insufficient C4 tokens: {len(tokens)}')
cases=[]
for n in (32,512,2048,8192,16384,32768,65536,130944):
 ids=tokens[:n];cases.append(dict(id=f'c4-concat:{n}',source='c4',example_id='validation-shard0-concatenation',split='heldout',category='text-continuation',prefix_tokens=n,token_ids=ids,prompt_sha256=prompt_digest([ids]),preprocessing='First documents in pinned English validation shard, concatenate text with two newlines, tokenize once without chat template, take first N tokens.'))
workload=dict(schema='akas.text-workload.v1',tokenizer_checkpoint_sha256='d1c03f73ecaf61017d70419eec9e5dc0b05d50ae26c23d522829b934fbabf8a5',sources=dict(c4=dict(repo='allenai/c4',revision=rev,file=file,url=url,documents=sources,concatenated_text_sha256=hashlib.sha256(text.encode()).hexdigest())),cases=cases,output_tokens=128,ignore_eos=True,scope='Quick single-user CUDA C4 text continuation sweep. One nested concatenated stream, eight prefixes, one measured sample per case; no training or corpus quality claim.')
write(out/'workload.json',workload);print('C4_READY',len(sources),'documents',len(tokens),'tokens',flush=True)
