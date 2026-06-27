#pragma once

#include <dl/autograd/Operator.hpp>

#include <vector>

namespace dl::autograd {

class AddOperator : public Operator {
public:
    std::vector<Tensor> backward(
        const std::vector<Tensor>& grad_outputs) override;
};

}  // namespace dl::autograd
