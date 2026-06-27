#pragma once

#include <dl/nn/Module.hpp>
#include <dl/tensor/Tensor.hpp>

namespace dl::nn {

class ReLU : public Module {
public:
    Tensor forward(const Tensor& input) override;
};

}  // namespace dl::nn
