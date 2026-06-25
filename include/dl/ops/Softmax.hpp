#pragma once

#include <cstddef>

#include <dl/tensor/Tensor.hpp>

namespace dl::ops {

::Tensor softmax(
    const ::Tensor& input,
    std::size_t rows,
    std::size_t cols);

}  // namespace dl::ops
