#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python3}"
BUILD_DIR="${BUILD_DIR:-build}"
ITERATIONS="${ITERATIONS:-100}"
WARMUP="${WARMUP:-10}"

echo "build : cpp ops benchmark"
cmake -S . -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD_DIR}" --target ops_benchmark

echo
echo "install : dl python api"
"${PYTHON_BIN}" -m pip install -e ".[test,benchmark]"

echo
echo "test : python api"
"${PYTHON_BIN}" -m pytest tests/python

echo
echo "benchmark : cpp dl ops"
"./${BUILD_DIR}/ops_benchmark"

echo
echo "benchmark : python dl ops"
"${PYTHON_BIN}" benchmarks/python/dl_ops_benchmark.py \
    --iterations "${ITERATIONS}" \
    --warmup "${WARMUP}"

echo
echo "benchmark : pytorch python ops"
"${PYTHON_BIN}" benchmarks/python/pytorch_ops_benchmark.py \
    --iterations "${ITERATIONS}" \
    --warmup "${WARMUP}"

echo
echo "compare : ops"
"${PYTHON_BIN}" benchmarks/python/compare_ops.py
