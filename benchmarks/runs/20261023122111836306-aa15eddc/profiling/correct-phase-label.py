"""Correct a diagnostic phase label while preserving its original raw report."""
from pathlib import Path
import json
from akas.benchmarking.history import digest,write,utc,rebuild_index
root=Path('/home/sagemaker-user/akas/benchmarks');folder=root/'runs/20261023122111836306-aa15eddc'
data=json.loads((folder/'run.json').read_text());report=json.loads((folder/'report.json').read_text())
if not report['metadata'].get('phase_label_correction'):
 original=folder/'profiling/report-before-label-correction.json';original.write_bytes((folder/'report.json').read_bytes())
 for case in report['metadata']['cases']:
  operations=case.get('operations',{})
  if 'proposal' in operations:operations['context']=operations.pop('proposal')
 correction=dict(kind='phase-label',reason='Post-repair draft.last_frame is the context frame. Correct only the per-operation group label; proposal/verify/commit/repair method timings and all numerical measurements remain unchanged.',original_report_sha256=digest(original),corrected_at=utc())
 report['metadata']['phase_label_correction']=correction;write(folder/'report.json',report);data['metadata']=report['metadata']
 data['artifacts'].append(dict(path='profiling/report-before-label-correction.json',sha256=digest(original),bytes=original.stat().st_size))
 destination=folder/'profiling/correct-phase-label.py';destination.write_bytes(Path(__file__).read_bytes());data['artifacts'].append(dict(path='profiling/correct-phase-label.py',sha256=digest(destination),bytes=destination.stat().st_size))
 for item in data['artifacts']:
  if item['path']=='report.json':item.update(sha256=digest(folder/'report.json'),bytes=(folder/'report.json').stat().st_size)
 write(folder/'run.json',data);rebuild_index(root)
 print('Corrected diagnostic operation label; original report retained')
