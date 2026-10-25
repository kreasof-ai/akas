#!/usr/bin/env bash
set -euo pipefail
cd /home/sagemaker-user/akas
source build/baselines/vulkan-env.sh
export CUDA_HOME=/usr/local/cuda-12.9
export PATH=/usr/local/cuda-12.9/bin:$PATH
export MAX_JOBS=2 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
.venv/bin/python - <<'PYCODE'
import json
from pathlib import Path
from akas.benchmarking.history import source_identity
path=Path('build/optimization-pilots/readiness.json')
if not path.exists():
    raise SystemExit('Readiness checks have not finished.')
report=json.loads(path.read_text())
if not report.get('bounded_supported_search_ready'):
    raise SystemExit('Readiness checks have unresolved failures; inspect '+str(path))
if report['source']['tree_sha256'] != source_identity(Path.cwd())['tree_sha256']:
    raise SystemExit('Source changed after preparation; repeat readiness checks.')
PYCODE
exec .venv/bin/python -m tools.campaign \
  --spec benchmarks/campaigns/standard-v1.json \
  --budget 4d --slice 30m --out build/standard-campaign
