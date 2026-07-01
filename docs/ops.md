# Ops

Ops are the public mathematical functions exposed under `dl::ops`.

The main header is:

```cpp
#include <dl/ops/Ops.hpp>
```

Most users include:

```cpp
#include <dl/dl.hpp>
```

## Design Rule

The user-facing op API should be simple:

```cpp
dl::Tensor y = dl::ops::softmax(x);
```

The user should not choose low-level kernel algorithms from the public API. Each op should internally use the best default implementation available.

## Current Public Ops

Current public ops:

```text
add
sub
mul
div
relu
leaky_relu
gelu
sigmoid
tanh
matmul
linear
softmax
cross_entropy
layernorm
```

Elementwise operators are also overloaded:

```cpp
Tensor c = a + b;
Tensor d = a - b;
Tensor e = a * b;
Tensor f = a / b;
```

## Op Implementation Pattern

Most ops follow this structure:

```text
1. validate inputs
2. allocate output tensor
3. launch CUDA kernel or vendor library call
4. check CUDA errors
5. attach autograd operator if needed
6. return output
```

Validation helpers live in:

```text
include/dl/ops/OpUtils.hpp
src/ops/OpUtils.cpp
```

Common validation helpers:

```cpp
check_defined(tensor, op_name);
check_rank(tensor, rank, op_name);
check_float32(tensor, op_name);
check_cuda(tensor, op_name);
check_same_shape(a, b, op_name);
check_same_dtype(a, b, op_name);
check_same_device(a, b, op_name);
check_binary_float_cuda_op(a, b, op_name);
```

## Elementwise Ops

Header:

```text
include/dl/ops/Elementwise.hpp
```

Source:

```text
src/ops/Elementwise.cpp
src/kernels/cuda/elementwise.cu
```

Supported ops:

```cpp
dl::ops::add(a, b);
dl::ops::sub(a, b);
dl::ops::mul(a, b);
dl::ops::div(a, b);
```

Input requirements:

```text
a and b are defined
same shape
same dtype
Float32
CUDA
same device
```

Output:

```text
same shape, dtype, and device as input
```

Current limitation:

No broadcasting yet.

## Activation Ops

Header:

```text
include/dl/ops/Activation.hpp
```

Source:

```text
src/ops/Activation.cpp
src/kernels/cuda/activation.cu
```

Supported forward ops:

```cpp
dl::ops::relu(x);
dl::ops::leaky_relu(x, 0.01f);
dl::ops::gelu(x);
dl::ops::sigmoid(x);
dl::ops::tanh(x);
```

Input requirements:

```text
defined
Float32
CUDA
```

Output:

```text
same shape, dtype, and device as input
```

Current autograd support:

```text
relu has backward support
leaky_relu, gelu, sigmoid, tanh currently have forward support only
```

## Matmul

Header:

```text
include/dl/ops/Matmul.hpp
```

Source:

```text
src/ops/Matmul.cpp
src/kernels/cuda/gemm.cu
```

API:

```cpp
dl::Tensor y = dl::ops::matmul(a, b);
```

Shape rule:

```text
a: [m, k]
b: [k, n]
y: [m, n]
```

Current implementation uses the GEMM kernel API:

```cpp
dl::kernels::gemm(...)
```

The active GEMM implementation is cuBLAS-backed.

Backward:

```text
grad_a = grad_y @ b^T
grad_b = a^T @ grad_y
```

## Linear

Header:

```text
include/dl/ops/Linear.hpp
```

Source:

```text
src/ops/Linear.cpp
src/kernels/cuda/bias.cu
```

API:

```cpp
dl::Tensor y = dl::ops::linear(input, weight, bias);
```

Shape rule:

```text
input:  [batch, in_features]
weight: [in_features, out_features]
bias:   [out_features]
output: [batch, out_features]
```

Implementation:

```text
matmul(input, weight)
add row bias
```

`linear` is an op because it is a stateless mathematical operation. `nn::Linear` is a module because it owns parameters.

## Softmax

Header:

```text
include/dl/ops/Softmax.hpp
```

Source:

```text
src/ops/Softmax.cpp
src/kernels/cuda/softmax.cu
```

API:

```cpp
dl::Tensor y = dl::ops::softmax(x);
```

Shape rule:

```text
x: [rows, cols]
y: [rows, cols]
```

Softmax is applied along the last dimension for a 2D tensor.

Current requirements:

```text
rank 2
Float32
CUDA
non-empty rows and cols
```

Current note:

The public API does not expose algorithm selection. The op calls the default kernel path.

## Cross Entropy

Header:

```text
include/dl/ops/CrossEntropy.hpp
```

Source:

```text
src/ops/CrossEntropy.cpp
src/kernels/cuda/cross_entropy.cu
```

API:

```cpp
dl::Tensor loss = dl::ops::cross_entropy(logits, labels);
```

Shape rule:

```text
logits: [batch, classes]  Float32
labels: [batch]           Int64
loss:   [1]               Float32
```

This is multiclass cross entropy, not binary cross entropy.

Backward:

```text
returns gradient for logits
returns empty Tensor for labels
```

Labels are integer class indices and do not receive gradients.

## LayerNorm

Header:

```text
include/dl/ops/LayerNorm.hpp
```

Source:

```text
src/ops/LayerNorm.cpp
src/kernels/cuda/layernorm.cu
```

API:

```cpp
dl::Tensor y = dl::ops::layernorm(x);
dl::Tensor y = dl::ops::layernorm(x, 1.0e-5f);
```

Shape rule:

```text
x: [rows, cols]
y: [rows, cols]
```

LayerNorm normalizes each row across `cols`.

Current requirements:

```text
rank 2
Float32
CUDA
```

Backward:

```text
returns gradient for input
```

Current limitation:

No learnable `gamma` or `beta` parameters in the public op yet.

## Ops And Autograd

Ops connect to autograd only when:

```text
gradient mode is enabled
at least one differentiable input requires grad
```

Example:

```cpp
dl::Tensor y = dl::ops::matmul(a, b);
```

If `a` or `b` requires grad:

```text
MatmulOperator is created
inputs are saved
output creator is set
output requires_grad becomes true
```

During backward, `MatmulOperator::backward()` computes gradients for each saved input.

## Ops And Modules

Ops are stateless functions.

Modules own parameters and call ops.

Example:

```text
dl::ops::linear(input, weight, bias)
```

is a function.

```text
dl::nn::Linear
```

owns:

```text
weight
bias
```

and calls the linear op inside its `forward()`.

## Benchmark Coverage

The op benchmark currently includes:

```text
add
sub
mul
div
relu
leaky_relu
gelu
sigmoid
tanh
matmul
linear
softmax
layernorm
cross_entropy
```

Run:

```bash
VENV_DIR=python-env bash scripts/compare_pytorch_ops.sh
```

The result reports:

```text
speedup = pytorch_avg_ms / dl_avg_ms
```

## Adding A New Op

Recommended checklist:

1. Add public declaration in `include/dl/ops/MyOp.hpp`.
2. Include it from `include/dl/ops/Ops.hpp`.
3. Add implementation in `src/ops/MyOp.cpp`.
4. Add validation using `OpUtils`.
5. Add kernel declaration in `include/dl/kernels/my_op.hpp`.
6. Add CUDA implementation in `src/kernels/cuda/my_op.cu`.
7. Add the new source files to `CMakeLists.txt`.
8. Add tensor correctness test in `tests/tensor/`.
9. If differentiable, add autograd operator in `include/dl/autograd/` and `src/autograd/`.
10. Add autograd test in `tests/autograd/`.
11. Add benchmark row in `tests/benchmarks/ops/ops_benchmark.cpp`.
12. Add PyTorch comparison row in `tests/benchmarks/pytorch/pytorch_ops_benchmark.py`.

## Current Limitations

- Most ops are CUDA-only.
- Most ops are `Float32` only.
- Broadcasting is not implemented.
- Many ops assume contiguous dense storage.
- Many ops only support rank 1 or rank 2.
- Activation backward support is incomplete.
- No `out=` public API yet for preallocated outputs.

## Next Ops Milestones

For CIFAR-10, the next important ops are:

```text
conv2d
maxpool2d
flatten
batchnorm or groupnorm
global average pooling
```

For Transformer demos, the next important ops are:

```text
embedding
transpose/view helpers
batched matmul
masked softmax
attention
dropout
```
