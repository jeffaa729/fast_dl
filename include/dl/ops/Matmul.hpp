#pragma once

#include <cstddef>

#include <dl/tensor/Tensor.hpp>

namespace dl::ops {

Tensor matmul(const Tensor& a, const Tensor& b);

}  // namespace dl::ops
