#pragma once

#include <cstddef>

namespace dl::kernels {

enum class ElementwiseOp {
    Add,
    Sub,
    Mul,
    Div,
};

const char* to_string(ElementwiseOp op);

void elementwise(const float* a, const float* b, float* c, int n, ElementwiseOp op);

}  // namespace dl::kernels
