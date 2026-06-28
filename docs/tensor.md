# Tensor

`dl::Tensor` is the main user-facing data object in the library. It represents a multidimensional array stored on a device, currently CUDA-first.

The public header is:

```cpp
#include <dl/tensor/Tensor.hpp>
```

Most users include:

```cpp
#include <dl/dl.hpp>
```

## Core Types

Tensor construction uses three small core types:

```text
Shape   tensor dimensions
DType   scalar type
Device  where the tensor lives
```

Example:

```cpp
dl::Tensor x(
    dl::Shape({2, 3}),
    dl::DType::Float32,
    dl::Device(dl::DeviceType::CUDA, 0));
```

This creates a CUDA tensor with:

```text
shape  = [2, 3]
dtype  = Float32
device = cuda:0
numel  = 6
nbytes = 24
```

## Device

`dl::Device` stores:

```cpp
DeviceType type;
int index;
```

Current device types:

```cpp
enum class DeviceType {
    CPU,
    CUDA
};
```

Current implementation is CUDA-first. Most tensor allocation and ops require:

```cpp
device.is_cuda()
```

CPU is present as a type, but CPU tensor execution is not the active backend yet.

## DType

Current dtype enum:

```cpp
enum class DType {
    Float32,
    Float16,
    Int32,
    Int64,
    Bool
};
```

`dtype_size(dtype)` returns the number of bytes per element.

Important current limitation:

Most math ops support `Float32` only. Labels for classification use `Int64`.

Typical usage:

```text
input images    Float32
weights         Float32
activations     Float32
logits          Float32
labels          Int64
```

Adding real `Float16`, `Int32`, or `Bool` op support requires matching kernels and validation logic.

## Shape

`dl::Shape` stores a vector of dimensions:

```cpp
std::vector<int64_t> dims;
```

Important methods:

```cpp
int64_t rank() const;
int64_t numel() const;
int64_t operator[](int index) const;
std::string str() const;
```

Examples:

```cpp
dl::Shape a({64, 784});       // rank 2, numel 50176
dl::Shape b({64, 3, 32, 32}); // rank 4, numel 196608
```

Tensor storage does not prevent 4D shapes. The limitation is that many current ops only implement 1D or 2D behavior.

## Tensor And TensorImpl

`Tensor` is a small value type:

```cpp
class Tensor {
    std::shared_ptr<TensorImpl> impl_;
};
```

Copying a `Tensor` copies the shared pointer, not the CUDA data.

`TensorImpl` owns the actual state:

```text
data_
shape_
dtype_
device_
requires_grad_
grad_
creator_
generation_
```

This split keeps the user API lightweight while keeping storage and autograd state in one implementation object.

## Construction Helpers

Current tensor creation helpers:

```cpp
Tensor::empty(shape, dtype, device);
Tensor::zeros(shape, dtype, device);
Tensor::randn(shape, dtype, device, mean, stddev, seed);
Tensor::uniform(shape, dtype, device, low, high, seed);
Tensor::xavier_uniform(shape, dtype, device, seed);
Tensor::kaiming_uniform(shape, dtype, device, seed);

Tensor::empty_like(other);
Tensor::zeros_like(other);
Tensor::ones_like(other);
Tensor::from_host<T>(data, shape, device);
```

Example:

```cpp
dl::Device device(dl::DeviceType::CUDA, 0);

dl::Tensor weights = dl::Tensor::kaiming_uniform(
    dl::Shape({784, 128}),
    dl::DType::Float32,
    device);

dl::Tensor bias = dl::Tensor::zeros(
    dl::Shape({128}),
    dl::DType::Float32,
    device);
```

Current random helpers support `Float32` CUDA tensors.

## Host Copy Helpers

Move data from CPU vectors into CUDA tensors:

```cpp
std::vector<float> values = {1.0f, 2.0f, 3.0f, 4.0f};

dl::Tensor x = dl::Tensor::from_host<float>(
    values,
    dl::Shape({2, 2}),
    device);
```

Move tensor data back to host:

```cpp
std::vector<float> host = x.to_host<float>();
```

Lower-level copy APIs:

```cpp
void copy_from_host(const void* src, std::size_t bytes);
void copy_to_host(void* dst, std::size_t bytes) const;
void copy_from(const Tensor& src);
void copy_to(Tensor& dst) const;
```

Current copy support:

```text
Host -> CUDA
CUDA -> Host
CUDA -> CUDA
```

## Memory Allocation

Tensor allocation is handled by:

```text
src/core/CudaAllocator.cpp
include/dl/core/CudaAllocator.hpp
```

`TensorImpl::allocate()` calls:

```cpp
dl::cuda::allocate(nbytes(), device_.index);
```

`TensorImpl::release()` calls:

```cpp
dl::cuda::deallocate(data_, nbytes(), device_.index);
```

This keeps raw CUDA allocation logic out of Tensor and makes it easier to improve allocation later.

## In-Place Zero

`zero_()` sets tensor memory to zero:

```cpp
x.zero_();
```

Current implementation uses `cudaMemset`, so it is suitable for zeroing numeric CUDA buffers.

## Autograd Fields

Tensor also stores autograd metadata through `TensorImpl`:

```cpp
bool requires_grad() const;
void set_requires_grad(bool value);

Tensor grad() const;
void zero_grad();

void backward();
void backward(const Tensor& grad);

int generation() const;
void set_creator(std::shared_ptr<Operator> creator);
std::shared_ptr<Operator> creator() const;
void accumulate_grad(const Tensor& grad);
```

Typical training tensor:

```cpp
dl::Tensor weight = dl::Tensor::kaiming_uniform(...);
weight.set_requires_grad(true);
```

`backward()` starts from a tensor that requires gradients and walks the dynamic computation graph.

## Value Semantics

`Tensor` behaves like a shared handle:

```cpp
dl::Tensor a = dl::Tensor::zeros(shape, dtype, device);
dl::Tensor b = a;
```

`a` and `b` refer to the same `TensorImpl`.

Use `copy_from()` if you want to copy data between two different allocated tensors:

```cpp
dl::Tensor dst = dl::Tensor::empty_like(src);
dst.copy_from(src);
```

## Current Limitations

- CUDA allocation only.
- Most ops support `Float32` only.
- No strides yet.
- No views yet.
- No broadcasting yet.
- Tensor shape supports N dimensions, but many ops only support 1D or 2D.
- `ones_like()` currently creates a host vector then copies to CUDA.

## Next Tensor Milestones

Good next improvements:

```text
strides and contiguous layout metadata
reshape/view support
4D NCHW convenience helpers
typed device fill kernels
CPU backend or explicit CUDA-only cleanup
better allocator statistics
out= APIs for benchmark-friendly preallocated outputs
```
