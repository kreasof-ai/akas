"""Archive original compiler study files without reclassifying their evidence."""
from pathlib import Path
from akas.benchmarking.history import Recorder
main=Path('/home/sagemaker-user/akas');source=main.parent/'tensor'
paths=[*sorted((source/'research/studies/compiler-design/evidence').glob('*.json')),source/'research/studies/compiler-design/reports/dynamic-cuda-pilot.md']
metadata=dict(runtime='tensor',purpose='historical-evidence-import',provider='cuda',model='generic-bounded-matmul',qualification_scope='Archive import of existing study files with their original measurement metadata; no rerun, current-source numerical qualification or hardware-ceiling claim')
rec=Recorder(main/'benchmarks',metadata,source_root=source,kind='diagnostic',change='Keep compiler benchmark evidence on artifact branches rather than the Tensor source branch')
rec.artifact(Path(__file__),'import.py')
for path in paths:
 assert path.is_file();rec.artifact(path,'original-evidence/'+path.name)
rec.update(dict(metadata=metadata,results=[]),final=True,status='passed')
print('ARCHIVE_IMPORT',rec.id,flush=True)
