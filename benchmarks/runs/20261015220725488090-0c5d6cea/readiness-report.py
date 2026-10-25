"""Consolidate retained qualification into one standard benchmark-history record."""
import json
from collections import Counter
from pathlib import Path
from akas.benchmarking.history import Recorder, write, digest, utc, source_identity
from akas.optimization.deployment import load
from akas.optimization.supervisor import verify_result

root = Path.cwd()
pilots = root / 'build/optimization-pilots'
selected = {}
for path in sorted((root / 'benchmarks/runs').glob('*/run.json')):
    record = json.loads(path.read_text())
    metadata = record['metadata']
    if metadata.get('protocol') != 'baseline-burst-v3' or metadata.get('purpose') != 'baseline' or len(record['results']) != 10:
        continue
    key = tuple(metadata[name] for name in ('runtime', 'model', 'provider', 'weight_format', 'mode'))
    selected[key] = (path, record)

allocations = [dict(path=str(path), sha256=digest(path), **json.loads(path.read_text())) for path in sorted(pilots.glob('allocation-*.json'))]
qpath = pilots / 'qwen-grouped-mtp-final/campaign.json'
qwen = json.loads(qpath.read_text())
pipeline_error = None
try:
    manifest = load(qpath.parent / 'deployment.json')
    for job in qwen['jobs']:
        if job['status'] == 'passed':
            verify_result(job)
    pipeline_passed = all(any(job['status'] == 'passed' and bool(job['recipe']['window']) == mtp for job in qwen['jobs']) for mtp in (False, True))
except Exception as error:
    pipeline_passed = False
    pipeline_error = f'{type(error).__name__}: {error}'
support = json.loads((pilots / 'matrix-support.json').read_text())

def measured(row):
    return len(row.get('samples', [])) == 3 and row.get('correctness', {}).get('output_count', False)

report = dict(
    schema='akas.readiness.v1', created_at=utc(), source=source_identity(root),
    full_requested_matrix_ready=False,
    limitations=[
        'Qwen WebGPU, Qwen FP8 KV and Qwen GGUF MTP are unimplemented; the matrix retains unsupported entries',
        'LFM BF16 source executes floating projections in FP16',
        'Large allocation checks do not execute every standard workload shape',
        'External MTP configurations that fail output qualification are excluded from qualified comparisons',
        'Baselines use optimized runtime defaults, not an exhaustive baseline tuning search',
        'Phase protocols differ: AKAS native kernels, native baseline request timings, llama.cpp server eval',
    ],
    baselines=[dict(
        key=list(key), path=str(path), id=record['id'], status=record['status'],
        shapes=len(record['results']), measured_shapes=sum(map(measured, record['results'])),
        passed_shapes=sum(row['status'] == 'ok' for row in record['results']),
        peak_device_bytes=record['metadata'].get('sampled_peak_device_bytes'),
        error=record['metadata'].get('error'),
    ) for key, (path, record) in selected.items()],
    allocations=allocations,
    qwen_pilot=dict(path=str(qpath), status=qwen['status'], error=pipeline_error,
        workload=json.loads((qpath.parent / 'deployment.json').read_text())['context']['workload'],
        jobs=[dict(status=job['status'], recipe=job['recipe']['id'], window=job['recipe']['window'], run_id=job.get('run_id'), error=job.get('error')) for job in qwen['jobs']]),
    support_inventory=dict(path=str(pilots / 'matrix-support.json'), scope=support['scope'], counts=dict(Counter(row['status'] for row in support['rows']))),
)
report['baseline_attempt_coverage_complete'] = len(selected) == 18
report['baseline_coverage_complete'] = len(selected) == 18 and all(all(map(measured, record['results'])) for _, record in selected.values())
report['baseline_qualification_complete'] = len(selected) == 18 and all(record['status'] == 'passed' for _, record in selected.values())
report['unqualified_baselines'] = [dict(id=record['id'], error=record['metadata'].get('error'), path=str(path)) for path, record in selected.values() if record['status'] != 'passed']
report['memory_allocations_passed'] = len(allocations) == 8 and all(row['status'] == 'passed' for row in allocations)
report['qwen_ar_mtp_pipeline_passed'] = pipeline_passed
report['regression_checks_passed'] = '666 passed, 694 skipped' in (pilots / 'full-suite-ready.log').read_text() and '2 passed' in (pilots / 'grouped-e2e.log').read_text()
report['source_unchanged'] = report['source']['tree_sha256'] == json.loads((pilots / 'preparation-source.json').read_text())['tree_sha256']
report['bounded_supported_search_ready'] = all(report[name] for name in ('baseline_coverage_complete', 'memory_allocations_passed', 'qwen_ar_mtp_pipeline_passed', 'regression_checks_passed', 'source_unchanged'))
write(pilots / 'readiness.json', report)
run = Recorder(root / 'benchmarks', dict(runtime='akas', purpose='readiness', protocol='standard-readiness-v1', model='standard-v1 matrix', provider='cuda+webgpu+vulkan', device=dict(name='NVIDIA L4'), coverage=report), kind='validation', change='Prepare bounded AKAS search; record measurement coverage and outstanding qualifications separately')
names = ['readiness.json', 'readiness-report.py', 'allocation-check.py', 'memory-check.json', 'spec-memory.json', 'full-suite-ready.log', 'grouped-e2e.log', 'group-geometry-tests.log', 'grouped-resume.log', 'interruption-grouped.log', 'qwen-grouped-mtp-final.log', 'q41-native-final.log', 'q41-consumers-final.log', 'matrix-budget-check.json', 'package-build.log', 'collector-final-tests.log', 'matrix-support.json', 'matrix-support.py', 'vllm-mtp-rerun.log', 'baseline-cache-fix.log', 'retry-qwen-baselines.py', 'check-archive.py', 'host-memory-diagnosis.json', 'final-checks.log', 'preparation-source.json', 'check-archive.log', 'external-mtp-agreement.json', 'campaign-list.json']
for path in [pilots / name for name in names] + [root / 'build/baselines/full-coverage.json', root / 'build/baselines/final-coverage.json', root / 'build/run-standard-campaign.sh', root / 'build/baselines/vulkan-env.sh']:
    if path.exists():
        run.artifact(path, path.name)
for row in allocations:
    run.artifact(Path(row['path']), 'allocations/' + Path(row['path']).name)
record = run.update(dict(schema='akas.benchmark.v1', metadata=run.record['metadata'], results=[]), final=True, status='passed' if report['bounded_supported_search_ready'] else 'failed')
print(record['id'], record['status'])
print(json.dumps({key: value for key, value in report.items() if isinstance(value, bool)}, indent=2))
