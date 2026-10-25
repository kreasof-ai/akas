import argparse,json
from pathlib import Path
from akas.benchmarking.history import Recorder,write,digest
p=argparse.ArgumentParser();p.add_argument('--program',type=Path,required=True);a=p.parse_args()
root=Path('build/vulkan-next')
meta=dict(runtime='akas',purpose='tuning-evidence',model='LiquidAI/LFM2.5-2.6B',provider='webgpu',target='webgpu-portable-v1',weight_format='q4_0',checkpoint_sha256='e1a61bf937bc60726e18626e97f7ee9bfd2574d95744c2ed909de98b78006fbe',scope='Collection of qualified and rejected kernel trials, synthetic Gaussian-cache decoding screens, isolated hot-buffer timings, changed-arithmetic regression checks, and retained real-text pilots. This collection creates no new inference samples. Partial/interrupted searches and failures are included and are not claims of passing candidates. The selected program is separately independently qualified and benchmarked on retained C4 token IDs.',selected_program_sha256=digest(a.program/'program.json'),reference_full_run_id='20261024215325218226-fd38f777',pilot_run_ids=['20261025100804512563-5b511685','20261025102646690468-5926d4a6','20261025102713303278-3fe0a083','20261025104117614877-c82e9c76','20261025115128345576-fd1a3267'],queue_control_run_ids=['20261025104200757861-c70bd924','20261025104227893571-17044c3e'],notes=['decode-grid-v3/results.json legacy granule field records the physical split count; the emitted kernel source fixes actual granule=128 and contains the true splits parameter. Corrected results are separate, preserving raw data.','Rejected streamed and native_f32 prototype helpers were removed from production; frozen helpers are under rejected-source. Other rejected prototypes retain compiled shader artifacts for replay; rebuilding every rejected historical variant from the final source is not claimed.'])
r=Recorder('benchmarks',meta,kind='preparation',parent='20261024215325218226-fd38f777',change='Retain context-wide Vulkan optimization evidence; no new performance samples')
for path in sorted(root.rglob('*')):
 if path.is_file() and 'debugtools' not in path.parts and path.suffix in ('.json','.py','.log','.md','.wgsl','.npz','.tbin','.crate','.txt','.rs') and path.name!='evidence.json':
  r.artifact(path,'experiments/'+str(path.relative_to(root)).replace('\\','/'))
for path in sorted(Path('build').glob('vulkan-next*.log')):r.artifact(path,'logs/'+path.name)
for path in sorted(Path('build').glob('vulkan-next*.py')):r.artifact(path,'replay/'+path.name)
for name in ('context_sweep','vulkan_tune','vulkan_profile','vulkan_decode_probe','vulkan_quality','vulkan_link'):
 r.artifact('tools/'+name+'.py','tools/'+name+'.py')
r.artifact(__file__,'archive.py');r.program(a.program)
saved=r.update(dict(metadata=meta,results=[]),final=True,status='passed')
write(root/'evidence.json',dict(run_id=r.id,status=saved['status']));print(r.id,flush=True)
