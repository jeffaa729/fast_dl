#pragma once

#include <dl/autograd/Operator.hpp>

#include <stdexcept>
#include <vector>

namespace dl::autograd {

class SubOperator : public Operator {
public:
    std::vector<Tensor> backward(
        const std::vector<Tensor>& grad_outputs) override;
};

}  // namespace dl::autograd
