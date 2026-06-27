#pragma once

#include <dl/tensor/Tensor.hpp>

#include <vector>

namespace dl::nn {

class Module {
public:
    virtual ~Module() = default;

    virtual Tensor forward(const Tensor& input) = 0;

    virtual std::vector<Tensor*> parameters() {
        return {};
    }

    Tensor operator()(const Tensor& input) {
        return forward(input);
    }
};

}  // namespace dl::nn