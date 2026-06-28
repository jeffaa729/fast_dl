# Getting Started

This project is a small C++ GPU-accelerated deep learning framework built on CUDA. The current workflow is Linux/WSL first, with CMake used for builds, CTest used for tests, and shell scripts used for benchmark runs.

## Requirements

- Linux or WSL with an NVIDIA GPU
- CMake 3.16 or newer
- A C++17 compiler
- CUDA Toolkit with `nvcc`
- NVIDIA driver visible from Linux
- Optional: Python venv with PyTorch for comparison benchmarks

Check CUDA visibility:

```bash
nvidia-smi
nvcc --version
```

## Configure And Build

From the repo root:

```bash
cmake -S . -B build
cmake --build build
```

Build one target:

```bash
cmake --build build --target tensor_test
cmake --build build --target mnist_mlp_demo
cmake --build build --target ops_benchmark
```

If CMake cannot detect the GPU architecture, pass it manually:

```bash
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=86
```

For an RTX 4060, `89` or `86` may be appropriate depending on the installed CUDA toolkit support.

## Run Tests

Run all registered tests:

```bash
ctest --test-dir build --output-on-failure
```

Run a single test executable:

```bash
./build/tensor_test
./build/softmax_tensor_test
./build/autograd_matmul_test
```

Tests should print a simple pass/fail status such as:

```text
tensor_test : passed
```

## Run The MNIST Demo

The MNIST demo expects IDX files in the repo root under `mnist/`:

```text
mnist/train-images-idx3-ubyte
mnist/train-labels-idx1-ubyte
mnist/t10k-images-idx3-ubyte
mnist/t10k-labels-idx1-ubyte
```

Build and run:

```bash
cmake --build build --target mnist_mlp_demo
./build/mnist_mlp_demo
```

Useful environment variables:

```bash
MNIST_EPOCHS=1 ./build/mnist_mlp_demo
MNIST_BATCH_SIZE=128 ./build/mnist_mlp_demo
MNIST_MAX_TRAIN_BATCHES=10 MNIST_MAX_TEST_BATCHES=5 ./build/mnist_mlp_demo
MNIST_LINEAR_INIT=xavier ./build/mnist_mlp_demo
```

Default initialization is Kaiming uniform.

## Run Op Benchmarks

Run the native C++/CUDA op benchmark:

```bash
cmake --build build --target ops_benchmark
./build/ops_benchmark
```

The result CSV is written to:

```text
benchmark_results/ops.csv
```

Configure warmup and timed iterations:

```bash
OPS_BENCH_WARMUP=10 OPS_BENCH_ITERS=100 ./build/ops_benchmark
```

## Compare Ops Against PyTorch

Create and activate a Python venv:

```bash
python3 -m venv python-env
source python-env/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Run the comparison:

```bash
VENV_DIR=python-env bash scripts/compare_pytorch_ops.sh
```

This runs:

- `tests/benchmarks/ops/ops_benchmark.cpp`
- `tests/benchmarks/pytorch/pytorch_ops_benchmark.py`
- `tests/benchmarks/compare/compare_ops.py`

Output CSV files:

```text
benchmark_results/ops.csv
benchmark_results/pytorch_ops.csv
benchmark_results/compare_ops.csv
```

The comparison uses:

```text
speedup = pytorch_avg_ms / dl_avg_ms
```

So `2.0x` means this library is 2x faster for that benchmark row.

## Compare MNIST Training Against PyTorch

Run a quick smoke test:

```bash
VENV_DIR=python-env MNIST_EPOCHS=1 MNIST_MAX_TRAIN_BATCHES=10 MNIST_MAX_TEST_BATCHES=5 bash scripts/compare_pytorch_mnist_mlp.sh
```

Run the full comparison:

```bash
VENV_DIR=python-env bash scripts/compare_pytorch_mnist_mlp.sh
```

Output CSV files:

```text
benchmark_results/dl_mnist_mlp_training.csv
benchmark_results/pytorch_mnist_mlp_training.csv
benchmark_results/compare_mnist_mlp_training.csv
```

## Nsight Profiling

Use Nsight Systems for end-to-end training behavior:

```bash
MNIST_EPOCHS=1 MNIST_MAX_TRAIN_BATCHES=20 MNIST_MAX_TEST_BATCHES=5 \
nsys profile --force-overwrite=true --trace=cuda,nvtx,osrt -o mnist_profile \
./build/mnist_mlp_demo
```

Use Nsight Compute for individual kernel metrics:

```bash
ncu ./build/ops_benchmark
```

As a rule:

- Use `nsys` to find where time goes across the full program.
- Use `ncu` to optimize one specific kernel.

## Current Limitations

- Most public ops currently support CUDA tensors only.
- Most math kernels currently support `Float32` only.
- The public op benchmark includes output allocation, so it is a user-facing op benchmark, not a pure kernel-only benchmark.
- Convolution, pooling, model save/load, mixed precision, and CUDA Graph training are future milestones.
