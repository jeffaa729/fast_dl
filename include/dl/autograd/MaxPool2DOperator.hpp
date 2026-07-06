#pragma once

#include <dl/autograd/Operator.hpp>

#include <vector>

namespace dl::autograd {

class MaxPool2DOperator : public Operator {
public:
    MaxPool2DOperator(int kernel_size, int stride, int padding);

    std::vector<Tensor> backward(
        const std::vector<Tensor>& grad_outputs) override;

private:
    int kernel_size_;
    int stride_;
    int padding_;
};

}  // namespace dl::autograd
