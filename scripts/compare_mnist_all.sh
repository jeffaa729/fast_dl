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

export MNIST_EPOCHS
export MNIST_BATCH_SIZE
export MNIST_MAX_TRAIN_BATCHES
export MNIST_MAX_TEST_BATCHES
export MNIST_TRAIN_EVAL_BATCHES
export MNIST_LR
export MNIST_LINEAR_INIT
export CUDACXX="${CUDACXX:-${CUDA_HOME}/bin/nvcc}"

echo "python : sync"
"${UV_BIN}" sync --extra benchmark

echo
echo "build : cpp mnist demo"
cmake -S . -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD_DIR}" --target mnist_mlp_demo

echo
echo "benchmark : dl python mnist mlp"
python_time=$(
    /usr/bin/time -f "python_dl_wall_s,%e" \
        "${UV_BIN}" run python examples/mnist_mlp.py 2>&1
)
echo "${python_time}"
python_wall_s="$(echo "${python_time}" | awk -F, '/python_dl_wall_s/ {print $2}')"

echo
echo "benchmark : dl cpp mnist mlp"
cpp_time=$(
    /usr/bin/time -f "cpp_dl_wall_s,%e" \
        "./${BUILD_DIR}/mnist_mlp_demo" 2>&1
)
echo "${cpp_time}"
cpp_wall_s="$(echo "${cpp_time}" | awk -F, '/cpp_dl_wall_s/ {print $2}')"

echo
echo "benchmark : pytorch mnist mlp"
torch_time=$(
    /usr/bin/time -f "pytorch_wall_s,%e" \
        "${UV_BIN}" run python tests/benchmarks/pytorch/pytorch_mnist_mlp_benchmark.py 2>&1
)
echo "${torch_time}"
pytorch_wall_s="$(echo "${torch_time}" | awk -F, '/pytorch_wall_s/ {print $2}')"

echo
echo "compare : mnist all"
"${UV_BIN}" run python - "${python_wall_s}" "${cpp_wall_s}" "${pytorch_wall_s}" <<'PY'
import sys

python_dl = float(sys.argv[1])
cpp_dl = float(sys.argv[2])
pytorch = float(sys.argv[3])

print(f"{'runner':<16}{'wall_s':>12}{'vs_cpp_dl':>12}{'vs_pytorch':>14}")
print("-" * 54)
for name, value in [
    ("dl_python", python_dl),
    ("dl_cpp", cpp_dl),
    ("pytorch", pytorch),
]:
    vs_cpp = value / cpp_dl if cpp_dl > 0 else float("inf")
    vs_torch = value / pytorch if pytorch > 0 else float("inf")
    print(f"{name:<16}{value:>12.2f}{vs_cpp:>11.2f}x{vs_torch:>13.2f}x")
PY
