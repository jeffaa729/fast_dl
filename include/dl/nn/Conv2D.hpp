#pragma once

#include <dl/nn/Module.hpp>
#include <dl/tensor/Tensor.hpp>

#include <vector>

namespace dl::nn {

enum class Conv2DInit {
    XavierUniform,
    KaimingUniform,
};

class Conv2D : public Module {
public:
    Conv2D(int in_channels,
           int out_channels,
           int kernel_size,
           const Device& device,
           int stride = 1,
           int padding = 0,
           Conv2DInit init = Conv2DInit::KaimingUniform);

    Tensor forward(const Tensor& input) override;

    std::vector<Tensor*> parameters() override;

private:
    Tensor weight_;
    Tensor bias_;
    int stride_;
    int padding_;
};

}  // namespace dl::nn
