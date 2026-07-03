#pragma once

#include <dl/autograd/Operator.hpp>

#include <vector>

namespace dl::autograd {

class FlattenOperator : public Operator {
public:

    explicit FlattenOperator(const Shape& input_shape);

    std::vector<Tensor> backward(
        const std::vector<Tensor>& grad_outputs) override;

private:
    Shape input_shape_;
};

}  // namespace dl::autograd
