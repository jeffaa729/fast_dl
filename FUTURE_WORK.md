# Future Work

This project is a CUDA-first deep learning library. The long-term goal is to grow from core tensor operations into full training and optimized inference demos.

## Proposed Architecture

The intended architecture is:

```text
User API
  -> Tensor / Module / Optimizer
  -> Ops
  -> Dispatcher or backend policy
  -> CUDA kernels / cuBLAS / fused kernels
```

Keep the active API CUDA-first and tensor-based. Old Eigen-based code should remain in `legacy/` unless it is rewritten around `dl::Tensor`.

### Core Tensor Architecture

```mermaid
classDiagram
    class Tensor {
        +empty(shape, dtype, device)
        +zeros(shape, dtype, device)
        +from_host(data, shape, device)
        +to_host()
        +data()
        +shape()
        +dtype()
        +device()
        +numel()
        +nbytes()
    }

    class TensorImpl {
        -void* data_
        -Shape shape_
        -DType dtype_
        -Device device_
        +copy_from_host()
        +copy_to_host()
        +zero_()
    }

    class Shape {
        +dims
        +rank()
        +numel()
    }

    class Device {
        +DeviceType type
        +int index
        +is_cuda()
    }

    class DType {
        <<enumeration>>
        Float32
        Float16
        Int32
        Int64
        Bool
    }

    Tensor --> TensorImpl
    TensorImpl --> Shape
    TensorImpl --> Device
    TensorImpl --> DType
```

### Ops And Kernel Path

```mermaid
flowchart TD
    A["User code: dl::ops::matmul(a, b)"] --> B["Validate shape, dtype, device"]
    B --> C["Allocate output Tensor"]
    C --> D["Select backend or algorithm"]
    D --> E["CUDA kernel or cuBLAS"]
    E --> F["Return output Tensor"]
```

Current ops should follow this structure:

```text
include/dl/ops/Foo.hpp
src/ops/Foo.cpp
include/dl/kernels/foo.hpp
src/kernels/cuda/foo.cu
tests/tensor/foo_tensor_test.cpp
```

### Future Autograd Architecture

```mermaid
classDiagram
    class Tensor {
        +backward()
        +grad()
        +requires_grad()
    }

    class TensorImpl {
        -data_
        -grad_
        -requires_grad_
        -grad_fn_
    }

    class Node {
        <<abstract>>
        +backward(grad_output)
    }

    class AddBackward
    class MatmulBackward
    class ReluBackward

    Tensor --> TensorImpl
    TensorImpl --> Node
    Node <|-- AddBackward
    Node <|-- MatmulBackward
    Node <|-- ReluBackward
```

Start autograd only after forward ops are stable. First support the minimum ops needed for MNIST MLP.

### Future Module And Training Architecture

```mermaid
classDiagram
    class Module {
        <<abstract>>
        +forward(input)
        +parameters()
        +train()
        +eval()
    }

    class Linear {
        -Tensor weight
        -Tensor bias
        +forward(input)
    }

    class ReLU {
        +forward(input)
    }

    class Sequential {
        -modules
        +forward(input)
        +parameters()
    }

    class Optimizer {
        <<abstract>>
        +step()
        +zero_grad()
    }

    class SGD
    class Adam

    Module <|-- Linear
    Module <|-- ReLU
    Module <|-- Sequential
    Optimizer <|-- SGD
    Optimizer <|-- Adam
    Optimizer --> Module
```

### Demo Dependency Roadmap

```mermaid
flowchart LR
    A["CUDA kernel benchmark suite"] --> B["MNIST MLP"]
    B --> C["CIFAR-10 ResNet"]
    B --> D["CIFAR-10 Vision Transformer"]
    D --> E["TinyStories GPT"]
    E --> F["GPT KV-cache inference"]
    A --> G["Mixed precision, fusion, CUDA Graphs"]
    E --> G
    F --> G
```

## Demo Roadmap

1. CUDA kernel benchmark suite
   - Keep benchmarking GEMM, softmax, reductions, transpose, elementwise ops, and future fused kernels.
   - Use Nsight Compute reports as performance evidence.

2. MNIST MLP
   - Prove basic tensor, layer, loss, optimizer, and autograd correctness.
   - Required pieces: `Linear`, `ReLU`, `matmul`, `add`, cross entropy, SGD, and backward passes.

3. CIFAR-10 ResNet
   - Add convolution support and train a small CNN/ResNet.
   - Required pieces: `Conv2D`, pooling, normalization, residual blocks, and image data loading.

4. CIFAR-10 Vision Transformer
   - Add Transformer encoder support for vision.
   - Required pieces: patch embedding, layer norm, attention, MLP block, and positional embeddings.

5. TinyStories GPT
   - Train a small GPT-style Transformer from scratch.
   - Required pieces: tokenization, embeddings, causal attention, Transformer blocks, optimizer state, and checkpointing.

6. GPT KV-cache inference
   - Add optimized autoregressive inference.
   - Required pieces: KV cache storage, incremental attention, fast sampling, and model loading.

7. Mixed precision, fusion, and CUDA Graphs
   - Improve framework performance after correctness is stable.
   - Required pieces: FP16/BF16 tensors, fused kernels, cuBLASLt, CUDA Graph capture, and async execution.

## Near-Term Engineering Tasks

1. Finish forward ops
   - Elementwise: `add`, `sub`, `mul`, `div`
   - Activations: `relu`, `sigmoid`, `tanh`
   - Neural network ops: `linear`, `cross_entropy`, `layernorm`

2. Add tests for every op
   - Each test should print:
     - `xxx_test : passed`
     - `xxx_test : not passed`

3. Add benchmarks for kernel-heavy ops
   - Keep `scripts/run_linux_benchmarks.sh` building all tests and benchmarks.
   - Keep Nsight Compute summaries readable.

4. Add autograd after forward ops are stable
   - Start with backward support for MNIST MLP only:
     - `add`
     - `matmul`
     - `relu`
     - `cross_entropy`

5. Add model/layer abstractions
   - Build layers on top of `dl::Tensor` and `dl::ops`.
   - Avoid bringing old Eigen-based code back into the active API.

6. Add checkpointing
   - Save and load model parameters as host-side binary tensor data.
   - Start with Float32 weights only.

7. Add inference features
   - `model.eval()`
   - fast model loading
   - KV cache for GPT-style generation
