# Autograd

Autograd is the automatic differentiation system. It records a dynamic computation graph during the forward pass, then walks that graph backward to compute gradients.

The main files are:

```text
include/dl/autograd/
src/autograd/
src/tensor/Tensor.cpp
src/tensor/TensorImpl.cpp
```

## Mental Model

The computation graph has two kinds of objects:

```text
TensorImpl  graph node, stores data and gradient state
Operator    graph edge, stores inputs and backward rule
```

Example:

```cpp
dl::Tensor z = x * y;
```

Forward pass:

```text
x, y -> MulOperator -> z
```

Backward pass:

```text
grad_z -> MulOperator::backward() -> grad_x, grad_y
```

## TensorImpl Autograd State

`TensorImpl` stores:

```text
requires_grad_  whether this tensor should receive gradients
grad_           accumulated gradient tensor
creator_        operator that produced this tensor
generation_     graph depth used for backward ordering
```

Leaf tensors created by the user usually have no creator:

```text
creator_ = nullptr
generation_ = 0
```

Parameters should set:

```cpp
weight.set_requires_grad(true);
bias.set_requires_grad(true);
```

## Operator State

The base operator stores:

```cpp
std::vector<Tensor> inputs_;
std::vector<std::weak_ptr<TensorImpl>> outputs_;
int generation_;
```

`inputs_` keeps the input tensors needed by backward.

`outputs_` uses `weak_ptr` so the graph does not create a memory cycle:

```text
Operator -> output TensorImpl -> creator Operator
```

If outputs were stored as `shared_ptr`, this cycle could keep graph objects alive forever.

## Building The Graph

Ops build the graph during forward.

The general pattern is:

```cpp
Tensor output = run_forward_kernel(...);

if (dl::autograd::is_grad_enabled() &&
    (input.requires_grad() || other.requires_grad())) {
    auto op = std::make_shared<SomeOperator>();
    op->setup_computation_graph({input, other}, {output});
    output.set_requires_grad(true);
    output.set_creator(op);
}

return output;
```

`setup_computation_graph()` does three things:

```text
save inputs
save weak references to outputs
compute operator generation
```

`set_creator()` connects the output tensor back to its creator and sets:

```text
output.generation = creator.generation + 1
```

## requires_grad Propagation

Rule:

```text
If any input requires grad, the output requires grad.
```

Example:

```cpp
x.set_requires_grad(true);
y.set_requires_grad(false);

z = x * y;
```

Then:

```text
z.requires_grad() == true
```

During backward, only inputs with `requires_grad() == true` receive accumulated gradients.

## NoGradGuard

`NoGradGuard` disables graph construction in a scope.

Example:

```cpp
{
    dl::autograd::NoGradGuard no_grad;
    dl::Tensor y = model(x);
}
```

This is used for:

```text
evaluation
benchmarks
internal backward calculations
```

Backward itself uses `NoGradGuard` so gradient formulas do not build a second graph.

## Backward Flow

`Tensor::backward()` is implemented in `src/tensor/Tensor.cpp`.

Calling:

```cpp
loss.backward();
```

does:

```text
1. create ones_like(loss) as the root gradient
2. accumulate root gradient into loss.grad
3. push loss.creator into a priority queue
4. pop highest-generation operator
5. collect grad_outputs from operator outputs
6. call op->backward(grad_outputs)
7. accumulate returned gradients into op inputs
8. push input creators into the queue
9. repeat until queue is empty
```

The priority queue processes higher-generation operators first.

## generation_

`generation_` is a simple topological ordering value.

Rules:

```text
leaf tensor generation = 0
operator generation = max(input generations)
output tensor generation = creator generation + 1
```

For a chain:

```text
x -> op1 -> y -> op2 -> z
```

Generations look like:

```text
x: 0
op1: 0
y: 1
op2: 1
z: 2
```

Backward starts from the highest generation and moves toward leaves.

## grad_outputs

Every backward operator receives:

```cpp
const std::vector<Tensor>& grad_outputs
```

For most single-output ops, this contains one tensor:

```text
grad_outputs[0] = gradient of the final loss with respect to this op's output
```

Example:

```cpp
z = x * y;
```

For `MulOperator::backward()`:

```text
grad_outputs[0] = dLoss/dz
inputs()[0]     = x
inputs()[1]     = y
```

The returned vector must match the number of inputs:

```text
return[0] = dLoss/dx
return[1] = dLoss/dy
```

## Current Backward Operators

Current autograd operators:

```text
AddOperator
SubOperator
MulOperator
DivOperator
ReLUOperator
MatmulOperator
AddRowBiasOperator
CrossEntropyOperator
LayerNormOperator
```

Elementwise rules:

```text
add: grad_a = grad_output
     grad_b = grad_output

sub: grad_a = grad_output
     grad_b = -grad_output

mul: grad_a = grad_output * b
     grad_b = grad_output * a

div: grad_a = grad_output / b
     grad_b = -(grad_output * a) / (b * b)
```

Matmul rules:

```text
C = A @ B

grad_A = grad_C @ B^T
grad_B = A^T @ grad_C
```

Cross entropy returns:

```text
grad_logits
empty gradient for labels
```

Labels do not require gradients.

## Gradient Accumulation

`TensorImpl::accumulate_grad()` does:

```text
if grad_ is empty:
    grad_ = incoming_grad
else:
    grad_ = grad_ + incoming_grad
```

This matters for graphs where one tensor contributes to multiple paths.

Example:

```cpp
y = x * x;
```

Both inputs refer to `x`, so gradients need to accumulate.

## Adding Autograd For A New Op

To add a new differentiable op:

1. Add public op declaration in `include/dl/ops/`.
2. Implement forward in `src/ops/`.
3. Add CUDA kernel launch API in `include/dl/kernels/`.
4. Implement CUDA kernel in `src/kernels/cuda/`.
5. Add an `Operator` subclass in `include/dl/autograd/`.
6. Implement `backward()` in `src/autograd/`.
7. In the forward op, create and attach the operator when grad is enabled.
8. Add tensor correctness tests.
9. Add autograd correctness tests.
10. Add benchmark rows if it is a public op.

Forward op graph connection example:

```cpp
if (dl::autograd::is_grad_enabled() && input.requires_grad()) {
    auto op = std::make_shared<MyOperator>();
    op->setup_computation_graph({input}, {output});
    output.set_requires_grad(true);
    output.set_creator(op);
}
```

## Current Limitations

- No broadcasting gradients.
- No detach API yet.
- No retain graph option.
- No gradient hooks.
- No higher-order gradients.
- Some activation ops have forward kernels but not backward operators yet.
- Backward formulas often allocate temporary tensors.

## Next Autograd Milestones

Useful next work:

```text
backward support for sigmoid, tanh, gelu, leaky_relu
Conv2D backward
MaxPool2D backward
Flatten backward
detach()
requires_grad initialization helpers for parameters
gradient checking utilities
better debug printing for graph structure
```
