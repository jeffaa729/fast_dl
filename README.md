# Deep Learning Library

A small C++ GPU-first neural-network framework used as the starting point for a higher-performance deep learning library. The current active build focuses on CUDA tensors, kernels, tests, and operation-specific benchmarks.

## Project Layout

```text
dl/
|-- include/dl/
|   |-- core/             Device, dtype, shape, errors
|   |-- tensor/           Tensor, storage, tensor views
|   |-- layers/           Neural-network layer abstractions
|   |-- ops/              Public operation APIs
|   |-- kernels/          Kernel launch declarations
|   |-- optim/            Optimizers
|   |-- runtime/          Network and execution runtime
|   `-- data/             Data loading
|-- src/
|   |-- core/
|   |-- tensor/
|   |-- layers/
|   |-- ops/
|   |-- kernels/
|   |   `-- cuda/         Future CUDA .cu implementations
|   |-- optim/
|   |-- runtime/
|   `-- data/
|-- tests/
|   |-- tensor/
|   |-- ops/
|   |-- layers/
|   |-- optim/
|   |-- runtime/
|   |-- data/
|   `-- benchmarks/
|       |-- gemm/
|       |-- softmax/
|       |-- transpose/
|       |-- reduction/
|       `-- vector_add/
|-- third_party/
|-- docs/
|-- mnist/
|-- CMakeLists.txt
`-- Makefile
```

## Build With CMake

```bash
cmake -S . -B build
cmake --build build
```

This GPU-first build requires the CUDA Toolkit.

To set the CUDA architecture explicitly:

```bash
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=86
```

## Build Targets

The core library target is:

```text
dl_core
```

Test/demo executables:

```text
tensor_test
dl_benchmarks
```

## Run Tests

```bash
ctest --test-dir build --output-on-failure
```

Run benchmarks:

```bash
./build/dl_benchmarks all
```

## Linux Test, Benchmark, And NCU Script

```bash
bash scripts/run_linux_benchmarks.sh
```

Optional overrides:

```bash
CUDA_ARCH=86 BENCHMARKS="gemm softmax" bash scripts/run_linux_benchmarks.sh
GEMM_N=1024 SOFTMAX_ROWS=4096 SOFTMAX_COLS=1024 bash scripts/run_linux_benchmarks.sh
```

The script builds `tensor_test` and `dl_benchmarks`, runs CTest, runs each benchmark, saves Nsight Compute reports under `ncu_reports/`, and writes selected CSV metrics.
It also prints a compact NCU summary with kernel runtime in microseconds plus DRAM, SM, L2, and L1 throughput as percent of peak.

## Legacy Make Commands

The root `Makefile` now delegates to CMake:

```bash
make
make test
make clean
```

## Future Direction

The next architecture step is to connect the CUDA kernels to higher-level tensor operations:

```text
core/device, core/dtype, core/shape
tensor/Tensor, tensor/Storage, tensor/Parameter
ops/gemm, ops/softmax, ops/attention, ops/layernorm
kernels/cuda/*.cu
tests/benchmarks/gemm, tests/benchmarks/softmax, tests/benchmarks/attention, tests/benchmarks/layernorm
```
