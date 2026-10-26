#!/bin/bash
set -e
export TENSOR_NVRTC_HOME="$PWD/build/nvrtc-12.9"
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export LD_LIBRARY_PATH="$PWD/build/baselines/cuda12-libs/nvidia/cublas/lib:$PWD/build/baselines/llama.cpp/build-cuda/bin:${LD_LIBRARY_PATH:-}"
spark_qualification_id=$(.venv/bin/python -c 'import json;print(json.load(open("build/spark-parity/qualification.json"))["run_id"])')
for spark_runtime in akas; do
 nvidia-smi --query-gpu=timestamp,name,temperature.gpu,power.draw,clocks.sm,clocks.mem,memory.used,utilization.gpu --format=csv --loop=1 > "build/spark-parity/gpu-${spark_runtime}.csv" &
 spark_monitor_pid=$!
 trap 'kill "$spark_monitor_pid" 2>/dev/null || true' EXIT
 .venv/bin/python -m tools.spark_benchmark "$spark_runtime" --program build/spark-parity/compact-program --chunk 2048 --qualification-run "$spark_qualification_id" --out "build/spark-parity/${spark_runtime}-full.json"
 kill "$spark_monitor_pid" 2>/dev/null || true
 wait "$spark_monitor_pid" 2>/dev/null || true
 trap - EXIT
done
