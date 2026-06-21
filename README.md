# Deep Learning Library

A small C++ neural-network framework used as the starting point for a higher-performance deep learning library. The current implementation uses Eigen as the CPU reference backend while the repo layout leaves room for future GPU tensors, kernels, and operation-specific benchmarks.

## Project Layout

```text
dl/
|-- include/dl/          Public headers
|-- src/                 Library implementation files
|-- tests/               Test and demo executables
|-- docs/                Project notes
|-- mnist/               MNIST data files
|-- images/              README/report images
|-- CMakeLists.txt       Main CMake build
`-- Makefile             Thin CMake wrapper
```

## Build With CMake

```bash
cmake -S . -B build
cmake --build build
```

The build first looks for Eigen in `./eigen-5.0.0`, then for a system `Eigen3` package. If neither exists, CMake can download Eigen 3.4.0 automatically through `FetchContent`.

To use a specific Eigen checkout:

```bash
cmake -S . -B build -DDL_EIGEN_INCLUDE_DIR=/path/to/eigen
```

To disable automatic Eigen download:

```bash
cmake -S . -B build -DDL_FETCH_EIGEN=OFF
```

## Build Targets

The core library target is:

```text
dl_core
```

Test/demo executables:

```text
test_layers
test_oop
test_network
test_train_mnist
```

## Run Tests

```bash
ctest --test-dir build --output-on-failure
```

For quick smoke testing, run:

```bash
ctest --test-dir build -R "test_layers|test_oop|test_network" --output-on-failure
```

`test_train_mnist` is also built by default, but it performs a longer MNIST training run.

## Legacy Make Commands

The root `Makefile` now delegates to CMake:

```bash
make
make test
make clean
```

## Future Direction

The next architecture step is to keep the current Eigen code as a CPU reference path while introducing GPU-oriented pieces:

```text
Tensor / Parameter
ops/gemm, ops/softmax, ops/attention
kernels/cuda
benchmarks/gemm, benchmarks/softmax, benchmarks/attention
```
