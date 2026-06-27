#pragma once

#include <cstddef>

#include <dl/tensor/Tensor.hpp>

namespace dl::ops {
    Tensor relu(const Tensor& a);
    Tensor leaky_relu(const Tensor& a, float alpha = 0.01f);
    Tensor gelu(const Tensor& a);
    Tensor sigmoid(const Tensor& a);
    Tensor tanh(const Tensor& a);
}