#pragma once

#include <vector>

#include <dl/tensor/Tensor.hpp>

namespace dl::optim {

class SGD {
public:
    SGD(const std::vector<Tensor*>& parameters, float learning_rate);

    void zero_grad();
    void step();

private:
    std::vector<Tensor*> parameters_;
    float learning_rate_;
};

}  // namespace dl::optim
