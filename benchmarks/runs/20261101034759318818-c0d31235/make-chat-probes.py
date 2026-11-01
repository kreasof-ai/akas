"""Freeze the checkpoint's own chat-template IDs and verify the common tokenizer."""
from pathlib import Path
import json,subprocess,hashlib
from akas.text.tokenizer import Tokenizer
from akas.benchmarking.history import Recorder

main=Path('/home/sagemaker-user/akas');root=main/'build/lfm-serving'
checkpoint=main/'build/checkpoints/lfm2.5-24b-a2b-fp8'
prompts=[
 'What is 17 + 25? Give the result as an integer.',
 'Sort the numbers 9, 2, 5 in ascending order.',
 'Extract the name and age as JSON: Ada is 37 years old.',
 'Write a Python function add(a, b) that returns their sum.',
 'The audit code in this message is DAHLIA-74. What is the audit code?',
 'All wugs are blue. Some blue things are small. Must every wug be small? Answer yes or no.',
 'Convert 3 hours to minutes.',
 'Return only the last letter of the word tensor.',
]
metadata=dict(runtime='tokenizer',purpose='chat-probe-inputs',model='LiquidAI/LFM2.5-24B-A2B-FP8',model_revision='f54bf39547fd5b7955abb9fe8d1b151e31fbbddb',qualification_scope='Chat input equivalence only; not model quality or throughput')
rec=Recorder(main/'benchmarks',metadata,source_root=main,kind='diagnostic',change='Freeze checkpoint chat-template probes with independent tokenizer ID agreement',dependency_sources={'tensor':main.parent/'tensor'})
rec.artifact(Path(__file__),'make-chat-probes.py')
try:
 worker='''import json,sys
from collections.abc import Mapping
from transformers import AutoTokenizer
p=json.load(sys.stdin)
t=AutoTokenizer.from_pretrained(p['checkpoint'],local_files_only=True)
rows=[]
for i,text in enumerate(p['prompts']):
 ids=t.apply_chat_template([{'role':'user','content':text}],tokenize=True,add_generation_prompt=True)
 if isinstance(ids,Mapping):ids=ids['input_ids']
 rows.append(dict(index=i,text=text,tokens=ids))
print(json.dumps(rows))
'''
 result=subprocess.run([str(main/'build/baselines/sglang/bin/python'),'-c',worker],input=json.dumps(dict(checkpoint=str(checkpoint),prompts=prompts)),text=True,capture_output=True)
 stderr=root/'chat-tokenizer-worker.stderr';stderr.write_text(result.stderr);rec.artifact(stderr,'worker.stderr')
 result.check_returncode()
 rows=json.loads(result.stdout)
 tokenizer=Tokenizer.from_pretrained(checkpoint)
 for row in rows:assert tokenizer.chat(row['text'])==row['tokens'],row['index']
 hashes={name:hashlib.sha256((checkpoint/name).read_bytes()).hexdigest() for name in ('chat_template.jinja','tokenizer.json','tokenizer_config.json')}
 out=root/'chat-probes.json';out.write_text(json.dumps(dict(model_revision=metadata['model_revision'],checkpoint_files=hashes,generated_tokens=256,greedy=True,ignore_eos=True,tokenizer_ids_equal=True,rows=rows),indent=2)+'\n')
 rec.artifact(out,'chat-probes.json')
 rec.update(dict(metadata=metadata,results=[]),final=True,status='passed');print('RUN',rec.id,flush=True)
except BaseException as error:
 rec.update(dict(metadata=metadata,results=[],error=repr(error)),final=True,status='failed');raise
