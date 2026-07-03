#pragma once

#include <dl/tensor/Tensor.hpp>

namespace dl::ops {
    Tensor conv2d(const Tensor& input,
                  const Tensor& weight,
                  const Tensor& bias,
                  int stride = 1,
                  int padding = 0);
}