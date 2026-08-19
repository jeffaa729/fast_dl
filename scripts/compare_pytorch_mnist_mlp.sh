#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${BUILD_DIR:-build}"
UV_BIN="${UV:-uv}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.3}"
MNIST_EPOCHS="${MNIST_EPOCHS:-20}"
MNIST_BATCH_SIZE="${MNIST_BATCH_SIZE:-64}"
MNIST_MAX_TRAIN_BATCHES="${MNIST_MAX_TRAIN_BATCHES:-0}"
MNIST_MAX_TEST_BATCHES="${MNIST_MAX_TEST_BATCHES:-0}"
MNIST_TRAIN_EVAL_BATCHES="${MNIST_TRAIN_EVAL_BATCHES:-0}"
MNIST_LR="${MNIST_LR:-0.01}"
MNIST_LINEAR_INIT="${MNIST_LINEAR_INIT:-kaiming}"
DL_MNIST_MLP_CSV="${DL_MNIST_MLP_CSV:-benchmark_results/dl_mnist_mlp_training.csv}"
PYTORCH_MNIST_MLP_CSV="${PYTORCH_MNIST_MLP_CSV:-benchmark_results/pytorch_mnist_mlp_training.csv}"
COMPARE_MNIST_MLP_CSV="${COMPARE_MNIST_MLP_CSV:-benchmark_results/compare_mnist_mlp_training.csv}"

export MNIST_EPOCHS
export MNIST_BATCH_SIZE
export MNIST_MAX_TRAIN_BATCHES
export MNIST_MAX_TEST_BATCHES
export MNIST_TRAIN_EVAL_BATCHES
export MNIST_LR
export MNIST_LINEAR_INIT
export DL_MNIST_MLP_CSV
export PYTORCH_MNIST_MLP_CSV
export COMPARE_MNIST_MLP_CSV
export CUDACXX="${CUDACXX:-${CUDA_HOME}/bin/nvcc}"

echo "python : sync"
"${UV_BIN}" sync --extra benchmark

echo "configure : ${BUILD_DIR}"
cmake -S . -B "${BUILD_DIR}"

echo
echo "build : mnist_mlp_demo"
cmake --build "${BUILD_DIR}" --target mnist_mlp_demo

echo
echo "benchmark : dl mnist mlp training"
mkdir -p "$(dirname "${DL_MNIST_MLP_CSV}")"
start_ns="$(date +%s%N)"
"${BUILD_DIR}/mnist_mlp_demo"
end_ns="$(date +%s%N)"
dl_total_ms="$("${UV_BIN}" run python - "${start_ns}" "${end_ns}" <<'PY'
import sys
start = int(sys.argv[1])
end = int(sys.argv[2])
print(f"{(end - start) / 1_000_000.0:.4f}")
PY
)"
dl_avg_epoch_ms="$("${UV_BIN}" run python - "${dl_total_ms}" "${MNIST_EPOCHS}" <<'PY'
import sys
total_ms = float(sys.argv[1])
epochs = int(sys.argv[2])
print(f"{total_ms / epochs:.4f}")
PY
)"
dl_train_batches="$("${UV_BIN}" run python - "${MNIST_MAX_TRAIN_BATCHES}" "${MNIST_BATCH_SIZE}" <<'PY'
import sys
max_batches = int(sys.argv[1])
batch_size = int(sys.argv[2])
print(max_batches if max_batches > 0 else (60000 + batch_size - 1) // batch_size)
PY
)"
cat > "${DL_MNIST_MLP_CSV}" <<CSV
name,epochs,batch_size,train_batches,total_ms,avg_epoch_ms
dl_mnist_mlp,${MNIST_EPOCHS},${MNIST_BATCH_SIZE},${dl_train_batches},${dl_total_ms},${dl_avg_epoch_ms}
CSV
echo "dl_mnist_mlp_total_ms : ${dl_total_ms}"
echo "dl_mnist_mlp_csv : ${DL_MNIST_MLP_CSV}"

echo
echo "benchmark : pytorch mnist mlp training"
"${UV_BIN}" run python tests/benchmarks/pytorch/pytorch_mnist_mlp_benchmark.py

echo
echo "compare : mnist mlp training"
"${UV_BIN}" run python tests/benchmarks/compare/compare_mnist_mlp_training.py
