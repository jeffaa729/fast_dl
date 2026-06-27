#pragma once

#include <dl/tensor/Tensor.hpp>

namespace dl::ops {

Tensor layernorm(const Tensor& input, float eps = 1.0e-5f);

}  // namespace dl::ops
