#include <cmath>
#include <cstdint>
#include <stdexcept>

#include <dl/nn/Linear.hpp>
#include <dl/ops/Linear.hpp>

namespace {

uint64_t next_linear_seed() {
    static uint64_t seed = 1234;
    return seed++;
}

}  // namespace

namespace dl::nn {

Linear::Linear(int in_features,
               int out_features,
               const Device& device,
               LinearInit init) {
    if (in_features <= 0 || out_features <= 0) {
        throw std::runtime_error("Linear features must be positive");
    }

    const Shape weight_shape({in_features, out_features});

    switch (init) {
        case LinearInit::XavierUniform:
            weight_ = Tensor::xavier_uniform(
                weight_shape,
                DType::Float32,
                device,
                next_linear_seed());
            break;
        case LinearInit::KaimingUniform:
            weight_ = Tensor::kaiming_uniform(
                weight_shape,
                DType::Float32,
                device,
                next_linear_seed());
            break;
    }

    bias_ = Tensor::zeros(
        Shape({out_features}),
        DType::Float32,
        device);
}

Tensor Linear::forward(const Tensor& input) {
    return dl::ops::linear(input, weight_, bias_);
}

std::vector<Tensor*> Linear::parameters() {
    return {&weight_, &bias_};
}

}  // namespace dl::nn
