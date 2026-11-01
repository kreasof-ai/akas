"""Preserve an interrupted source capture without inventing a benchmark run."""
from pathlib import Path
import shutil
from akas.benchmarking.history import Recorder
main=Path('/home/sagemaker-user/akas');source=main/'benchmarks/runs/20261101042846831193-019e2eb8'
metadata=dict(runtime='akas',purpose='incomplete-startup-evidence-import',original_directory=source.name,qualification_scope='Source snapshots from a recorder startup that never wrote run.json; no measurement, completion, source qualification or hardware-ceiling claim')
rec=Recorder(main/'benchmarks',metadata,source_root=main,kind='diagnostic',change='Retain interrupted recorder startup files separately from completed benchmark evidence',dependency_sources={'tensor':main.parent/'tensor'})
rec.artifact(Path(__file__),'import.py')
for path in source.rglob('*'):
 if path.is_file():rec.artifact(path,'incomplete-startup/'+str(path.relative_to(source)))
rec.update(dict(metadata=metadata,results=[]),final=True,status='passed')
retained=main/'build/incomplete-benchmark-startups'/source.name;retained.parent.mkdir(exist_ok=True)
assert not retained.exists();shutil.move(source,retained)
print('ARCHIVE_IMPORT',rec.id,'ORIGINAL_RETAINED',retained,flush=True)
