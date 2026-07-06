#pragma once

#include <dl/nn/Module.hpp>
#include <dl/tensor/Tensor.hpp>

namespace dl::nn {

class MaxPool2D : public Module {
public:
    explicit MaxPool2D(int kernel_size, int stride = -1, int padding = 0);

    Tensor forward(const Tensor& input) override;

private:
    int kernel_size_;
    int stride_;
    int padding_;
};

}  // namespace dl::nn
