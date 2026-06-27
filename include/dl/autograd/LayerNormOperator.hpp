#pragma once

#include <dl/autograd/Operator.hpp>

#include <vector>

namespace dl::autograd {

class LayerNormOperator : public Operator {
public:
    explicit LayerNormOperator(float eps);

    std::vector<Tensor> backward(
        const std::vector<Tensor>& grad_outputs) override;

private:
    float eps_;
};

}  // namespace dl::autograd
