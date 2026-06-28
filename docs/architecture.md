# Architecture

The library is organized around a small stack:

```text
User API
  Tensor, ops, nn::Module, optim::SGD

Core runtime
  Device, DType, Shape, TensorImpl, CUDA allocator

Autograd
  Operator graph, creator links, generation ordering, backward functions

Ops
  Public operation APIs such as matmul, softmax, layernorm, cross_entropy

Kernels
  CUDA implementations and vendor calls such as cuBLAS GEMM
```

The main public include is:

```cpp
#include <dl/dl.hpp>
```

## Folder Layout

```text
include/dl/core/       Device, DType, Shape, CUDA utilities
include/dl/tensor/     Tensor and TensorImpl declarations
include/dl/ops/        Public operation APIs
include/dl/autograd/   Operator classes for backward graph edges
include/dl/nn/         Neural network modules
include/dl/optim/      Optimizers
include/dl/kernels/    CUDA kernel launch APIs

src/core/              Runtime implementation
src/tensor/            Tensor and TensorImpl implementation
src/ops/               Public op implementation and validation
src/autograd/          Backward operator implementation
src/nn/                Module implementation
src/optim/             Optimizer implementation
src/kernels/cuda/      CUDA kernels and cuBLAS-backed kernels

tests/tensor/          Tensor and op correctness tests
tests/autograd/        Backward correctness tests
tests/nn/              Module and training-loop tests
tests/data/            Data loader tests
tests/demo/            Runnable demos
tests/benchmarks/      Native, PyTorch, and comparison benchmarks
scripts/               Build, benchmark, and profiling helpers
legacy/                Old experimental kernels and earlier code
```

## Tensor Layer

`dl::Tensor` is the public value type. It owns a `std::shared_ptr<TensorImpl>` and forwards most work to the implementation object.

Important public responsibilities:

- expose `shape()`, `dtype()`, `device()`
- expose `numel()` and `nbytes()`
- copy data from host and back to host
- hold autograd state through `requires_grad`, `grad`, and `creator`
- start backpropagation with `backward()`

Example:

```cpp
dl::Tensor x(
    dl::Shape({2, 3}),
    dl::DType::Float32,
    dl::Device(dl::DeviceType::CUDA, 0));
```

`TensorImpl` is the internal storage object. It owns:

- `data_`
- `shape_`
- `dtype_`
- `device_`
- `grad_`
- `creator_`
- `generation_`
- `requires_grad_`

CUDA memory allocation goes through `src/core/CudaAllocator.cpp`, rather than every tensor calling raw CUDA allocation code directly.

## Core Runtime

The core layer contains small types shared by the whole library:

```text
Device      where the tensor lives
DType       scalar type
Shape       dimensions and numel
CudaUtils   CUDA error checking
Allocator   CUDA memory allocation wrapper
```

Current design is CUDA-first. CPU tensors are not the main execution target yet.

## Ops Layer

Public operations live under `dl::ops`.

Current public ops include:

```text
add, sub, mul, div
relu, leaky_relu, gelu, sigmoid, tanh
matmul
linear
softmax
cross_entropy
layernorm
```

The public op layer is responsible for:

- checking input shape
- checking dtype
- checking device
- creating output tensors
- launching the correct kernel
- connecting autograd operators when gradient mode is enabled

For example, the rough flow for an op is:

```text
validate inputs
allocate output Tensor
launch CUDA kernel
if grad enabled:
    create backward Operator
    save inputs and outputs
    set output creator
return output
```

The user should not select low-level algorithms from the public op API. The op should choose the default fastest implementation available in the library.

## Kernel Layer

CUDA launch APIs are declared in:

```text
include/dl/kernels/
```

CUDA implementations live in:

```text
src/kernels/cuda/
```

This layer should stay focused on numerical work. It should not know about autograd, modules, or high-level model structure.

Current examples:

```text
elementwise.cu
activation.cu
gemm.cu
softmax.cu
cross_entropy.cu
layernorm.cu
optimizer.cu
random.cu
```

`gemm.cu` uses cuBLAS for matrix multiplication. Other ops use custom CUDA kernels.

## Autograd Layer

Autograd uses a dynamic computation graph. The graph is built during the forward pass.

The two key objects are:

```text
TensorImpl  graph node
Operator    graph edge
```

`TensorImpl` stores:

```text
grad_
creator_
generation_
requires_grad_
```

`Operator` stores:

```text
inputs_
outputs_
generation_
```

`outputs_` uses `weak_ptr<TensorImpl>` to avoid a reference cycle:

```text
Operator -> output TensorImpl -> creator Operator
```

Backward operators implement:

```cpp
virtual std::vector<Tensor> backward(
    const std::vector<Tensor>& grad_outputs) = 0;
```

Example:

```text
z = x * y
```

The multiply backward rule is:

```text
dz/dx = grad_output * y
dz/dy = grad_output * x
```

So `MulOperator::backward()` returns two tensors, one gradient for each saved input.

## Generation Ordering

`generation_` records graph depth.

Rules:

```text
leaf tensor generation = 0
operator generation = max(input generations)
output tensor generation = creator generation + 1
```

During backward, higher-generation operators should run before lower-generation operators. This gives a simple topological order without building a global static graph.

## NN Layer

The neural network API is modeled after PyTorch-style modules.

Base class:

```cpp
class Module {
public:
    virtual Tensor forward(const Tensor& input) = 0;
    Tensor operator()(const Tensor& input);
    virtual std::vector<Tensor*> parameters();
    void train();
    void eval();
};
```

Current modules:

```text
Linear
ReLU
Sequential
```

Custom models can inherit from `dl::nn::Module`:

```cpp
class MnistMLP : public dl::nn::Module {
public:
    MnistMLP(const dl::Device& device)
        : fc1_(784, 128, device),
          relu_(),
          fc2_(128, 10, device) {
        register_module(fc1_);
        register_module(relu_);
        register_module(fc2_);
    }

    dl::Tensor forward(const dl::Tensor& input) override {
        return fc2_(relu_(fc1_(input)));
    }

private:
    dl::nn::Linear fc1_;
    dl::nn::ReLU relu_;
    dl::nn::Linear fc2_;
};
```

`register_module()` allows parent modules to collect child parameters automatically.

## Optimizer Layer

The current optimizer is:

```text
dl::optim::SGD
```

It works over `std::vector<Tensor*>` from `Module::parameters()`.

Typical training step:

```cpp
optimizer.zero_grad();
dl::Tensor logits = model(images);
dl::Tensor loss = dl::ops::cross_entropy(logits, labels);
loss.backward();
optimizer.step();
```

## Demos

Runnable examples live in:

```text
tests/demo/
```

Current main demo:

```text
mnist_mlp_demo.cpp
```

This demo exercises:

- MNIST loading
- custom `Module`
- Linear
- ReLU
- CrossEntropy
- autograd
- SGD
- train/eval flow

Future demos should remain practical and runnable. Good milestones are:

```text
CIFAR-10 CNN
CIFAR-10 ResNet
CIFAR-10 Vision Transformer
TinyStories GPT
GPT KV-cache inference
```

## Benchmarks

There are two benchmark styles.

Op benchmarks compare public operations:

```bash
VENV_DIR=python-env bash scripts/compare_pytorch_ops.sh
```

Training benchmarks compare full demo training time:

```bash
VENV_DIR=python-env bash scripts/compare_pytorch_mnist_mlp.sh
```

The benchmark comparison scripts report:

```text
speedup = pytorch_time / dl_time
```

So `> 1.0x` means this library is faster for that row.

## Design Direction

The current design favors a simple CUDA-first training framework:

```text
Tensor owns CUDA memory
Ops hide kernel choices
Autograd builds a dynamic graph
Modules compose ops into models
Optimizers update tensor parameters
Benchmarks compare against PyTorch
```

The next architectural milestones are:

- 4D tensor workflows for vision
- CIFAR-10 data loading
- Conv2D forward and backward
- MaxPool2D and Flatten
- model save/load
- mixed precision
- CUDA Graph training capture
