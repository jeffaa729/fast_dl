#pragma once

#include <cstddef>

#include <dl/tensor/Tensor.hpp>

namespace dl::ops {
    Tensor add(const Tensor& a, const Tensor& b);
    Tensor sub(const Tensor& a, const Tensor& b);
    Tensor mul(const Tensor& a, const Tensor& b);
    Tensor div(const Tensor& a, const Tensor& b);
}

namespace dl {
    Tensor operator+(const Tensor& a, const Tensor& b);
    Tensor operator-(const Tensor& a, const Tensor& b);
    Tensor operator*(const Tensor& a, const Tensor& b);
    Tensor operator/(const Tensor& a, const Tensor& b);
}
