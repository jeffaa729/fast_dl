#pragma once

#include <cstddef>

namespace dl::kernels {

enum class ActivationOp {
    ReLU,
    LeakyReLU,
    GELU,
    Sigmoid,
    Tanh
};

const char* to_string(ActivationOp op);

void activation(const float* a, float* b, int n, ActivationOp op, float alpha = 0.01f);
void relu_backward(const float* input, const float* grad_output, float* grad_input, int n);

}  // namespace dl::kernels
