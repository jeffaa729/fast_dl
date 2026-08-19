from contextlib import contextmanager

from ._C import (
    Conv2D,
    Conv2DInit,
    DType,
    Device,
    DeviceType,
    Linear,
    LinearInit,
    MaxPool2D,
    Module,
    ReLU,
    SGD,
    Sequential,
    Shape,
    Tensor,
    add,
    bool,
    conv2d,
    cpu,
    cross_entropy,
    cuda,
    cuda_synchronize,
    div,
    flatten,
    float16,
    float32,
    gelu,
    int32,
    int64,
    is_grad_enabled,
    layer_norm,
    layernorm,
    leaky_relu,
    linear,
    matmul,
    max_pool2d,
    mul,
    relu,
    sigmoid,
    softmax,
    set_grad_enabled,
    sub,
    tanh,
)

__all__ = [
    "Conv2D",
    "Conv2DInit",
    "DType",
    "Device",
    "DeviceType",
    "Linear",
    "LinearInit",
    "MaxPool2D",
    "Module",
    "ReLU",
    "SGD",
    "Sequential",
    "Shape",
    "Tensor",
    "add",
    "bool",
    "conv2d",
    "cpu",
    "cross_entropy",
    "cuda",
    "cuda_synchronize",
    "div",
    "flatten",
    "float16",
    "float32",
    "gelu",
    "int32",
    "int64",
    "is_grad_enabled",
    "layer_norm",
    "layernorm",
    "leaky_relu",
    "linear",
    "matmul",
    "max_pool2d",
    "mul",
    "relu",
    "sigmoid",
    "softmax",
    "set_grad_enabled",
    "sub",
    "tanh",
    "no_grad",
]


@contextmanager
def no_grad():
    previous = is_grad_enabled()
    set_grad_enabled(False)
    try:
        yield
    finally:
        set_grad_enabled(previous)


def main() -> None:
    print("dl Python API is installed")
