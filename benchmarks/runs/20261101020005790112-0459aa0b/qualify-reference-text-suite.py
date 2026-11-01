"""Retain the independent original-weight natural-text reference generation."""
from pathlib import Path
import runpy
from akas.benchmarking.history import Recorder
root=Path('build/lfm-serving')
metadata=dict(runtime='numpy',purpose='numerical-reference',provider='cpu',model='LiquidAI/LFM2.5-24B-A2B-FP8',model_revision='f54bf39547fd5b7955abb9fe8d1b151e31fbbddb',qualification_scope='Independent F64 products with declared F32 boundaries for eight natural prompts; not serving throughput')
record=Recorder('benchmarks',metadata,kind='qualification',change='Regenerate independent natural-text probes after preserving F32 router operands',dependency_sources={'tensor':Path('../tensor')})
for path in (Path(__file__),root/'reference-text-suite.py'):record.artifact(path,path.name)
print('STARTED',record.id,flush=True)
try:
 runpy.run_path(str(root/'reference-text-suite.py'),run_name='__main__')
 for suffix in ('json','npz'):record.artifact(root/f'reference-text-suite.{suffix}',f'reference-text-suite.{suffix}')
 record.update(dict(metadata=metadata,results=[]),final=True,status='passed')
 print('RUN',record.id,flush=True)
except BaseException as error:
 if record.record['completed_at'] is None:record.update(dict(metadata=metadata,results=[],error=repr(error)),final=True,status='failed')
 raise
