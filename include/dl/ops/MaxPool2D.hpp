#pragma once

#include <dl/tensor/Tensor.hpp>

namespace dl::ops {

Tensor max_pool2d(const Tensor& input,
                  int kernel_size,
                  int stride = -1,
                  int padding = 0);

}  // namespace dl::ops
