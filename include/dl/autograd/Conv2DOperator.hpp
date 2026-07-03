#pragma once

#include <dl/autograd/Operator.hpp>

#include <vector>

namespace dl::autograd {

class Conv2DOperator : public Operator {
public:

    explicit Conv2DOperator(const Shape& input_shape, int stride, int padding);

    std::vector<Tensor> backward(
        const std::vector<Tensor>& grad_outputs) override;

private:
    Shape input_shape_;
    int stride_;
    int padding_;
};

}  // namespace dl::autograd
