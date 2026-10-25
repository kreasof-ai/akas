import json,tarfile,zipfile
from pathlib import Path
import jsonschema
from akas.benchmarking.history import digest
root=Path.cwd();schema=json.loads(Path('benchmarks/run.schema.json').read_text());count=artifacts=0
for path in sorted(Path('benchmarks/runs').glob('*/run.json')):
 record=json.loads(path.read_text())
 jsonschema.validate(record,schema)
 if record['status']=='running':continue
 for item in record['artifacts']:
  file=path.parent/item['path']
  assert file.stat().st_size==item['bytes'],file
  assert digest(file)==item['sha256'],file
  artifacts+=1
 count+=1
print('Validated',count,'sealed history records and',artifacts,'artifact checksums')
with tarfile.open('build/readiness-dist/akas-0.1.0.tar.gz') as archive:
 for name in ['tools/campaign.py','tools/baselines/llama.py','tools/baselines/common.py','benchmarks/models/qwen3.5-0.8b.json','benchmarks/campaigns/standard-v1.json','docs/optimization.md']:
  member=archive.extractfile('akas-0.1.0/'+name)
  assert member is not None and member.read()==Path(name).read_bytes(),name
with zipfile.ZipFile('build/readiness-dist/akas-0.1.0-py3-none-any.whl') as archive:
 for name in ['optimization/planning.py','optimization/worker.py','benchmarking/workloads/standard-v1.json']:
  assert archive.read('akas/'+name)==Path('src/akas',name).read_bytes(),name
print('Distribution contents match the working source')
