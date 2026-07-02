#pragma once

#include <cstddef>

#include <dl/tensor/Tensor.hpp>

namespace dl::ops {
    Tensor flatten(const Tensor& a);
}