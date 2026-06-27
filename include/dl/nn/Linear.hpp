#pragma once

#include <dl/nn/Module.hpp>
#include <dl/tensor/Tensor.hpp>

namespace dl::nn {

enum class LinearInit {
    XavierUniform,
    KaimingUniform,
};

class Linear : public Module {
public:
    Linear(int in_features,
           int out_features,
           const Device& device,
           LinearInit init = LinearInit::KaimingUniform);

    Tensor forward(const Tensor& input) override;

    std::vector<Tensor*> parameters() override;

private:
    Tensor weight_;
    Tensor bias_;
};

}  // namespace dl::nn
