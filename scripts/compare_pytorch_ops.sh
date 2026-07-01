#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${BUILD_DIR:-build}"
OPS_BENCH_WARMUP="${OPS_BENCH_WARMUP:-10}"
OPS_BENCH_ITERS="${OPS_BENCH_ITERS:-100}"
OPS_BENCH_CSV="${OPS_BENCH_CSV:-benchmark_results/ops.csv}"
PYTORCH_BENCH_CSV="${PYTORCH_BENCH_CSV:-benchmark_results/pytorch_ops.csv}"
COMPARE_BENCH_CSV="${COMPARE_BENCH_CSV:-benchmark_results/compare_ops.csv}"
VENV_DIR="${VENV_DIR:-.venv}"
LOG_DIR="${LOG_DIR:-benchmark_results}"

if [ -z "${PYTHON_CMD:-}" ]; then
    if [ -x "${VENV_DIR}/bin/python" ]; then
        PYTHON_CMD="${VENV_DIR}/bin/python"
    else
        PYTHON_CMD="python"
    fi
fi

export OPS_BENCH_WARMUP
export OPS_BENCH_ITERS
export OPS_BENCH_CSV
export PYTORCH_BENCH_CSV
export COMPARE_BENCH_CSV

mkdir -p "${LOG_DIR}"

echo "python_cmd : ${PYTHON_CMD}"
echo "configure : ${BUILD_DIR}"
cmake -S . -B "${BUILD_DIR}"

echo
echo "build : ops_benchmark"
cmake --build "${BUILD_DIR}" --target ops_benchmark

echo
echo "benchmark : dl ops"
"${BUILD_DIR}/ops_benchmark" > "${LOG_DIR}/ops_benchmark.log"
echo "dl_ops_log : ${LOG_DIR}/ops_benchmark.log"

echo
echo "benchmark : pytorch ops"
"${PYTHON_CMD}" tests/benchmarks/pytorch/pytorch_ops_benchmark.py > "${LOG_DIR}/pytorch_ops_benchmark.log"
echo "pytorch_ops_log : ${LOG_DIR}/pytorch_ops_benchmark.log"

echo
echo "compare : ops"
"${PYTHON_CMD}" tests/benchmarks/compare/compare_ops.py
