#!/usr/bin/env bash
set -euo pipefail

UV_BIN="${UV:-uv}"
BUILD_DIR="${BUILD_DIR:-build}"
ITERATIONS="${ITERATIONS:-100}"
WARMUP="${WARMUP:-10}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.3}"
export CUDACXX="${CUDACXX:-${CUDA_HOME}/bin/nvcc}"

echo "build : cpp ops benchmark"
cmake -S . -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD_DIR}" --target ops_benchmark

echo
echo "python : sync"
"${UV_BIN}" sync --extra test --extra benchmark

echo
echo "test : python api"
"${UV_BIN}" run pytest tests/python

echo
echo "benchmark : cpp dl ops"
"./${BUILD_DIR}/ops_benchmark"

echo
echo "benchmark : python dl ops"
"${UV_BIN}" run python benchmarks/python/dl_ops_benchmark.py \
    --iterations "${ITERATIONS}" \
    --warmup "${WARMUP}"

echo
echo "benchmark : pytorch python ops"
"${UV_BIN}" run python benchmarks/python/pytorch_ops_benchmark.py \
    --iterations "${ITERATIONS}" \
    --warmup "${WARMUP}"

echo
echo "compare : ops"
"${UV_BIN}" run python benchmarks/python/compare_ops.py
