#pragma once

#include <dl/tensor/Tensor.hpp>

namespace dl::ops {
    Tensor cross_entropy(const Tensor& logits, const Tensor& labels);
}