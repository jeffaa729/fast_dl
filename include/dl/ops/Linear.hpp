#pragma once

#include <dl/tensor/Tensor.hpp>

namespace dl::ops {
    Tensor linear(const Tensor& input, const Tensor& weight, const Tensor& bias);
}