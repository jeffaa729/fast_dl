#include <dl/nn/Conv2D.hpp>

#include <dl/ops/Conv2D.hpp>

#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace {

uint64_t next_conv2d_seed() {
    static uint64_t seed = 2234;
    return seed++;
}

}  // namespace

namespace dl::nn {

Conv2D::Conv2D(int in_channels,
               int out_channels,
               int kernel_size,
               const Device& device,
               int stride,
               int padding,
               Conv2DInit init)
    : stride_(stride),
      padding_(padding) {
    if (in_channels <= 0 || out_channels <= 0 || kernel_size <= 0) {
        throw std::runtime_error("Conv2D channels and kernel size must be positive");
    }
    if (stride <= 0 || padding < 0) {
        throw std::runtime_error("Conv2D stride must be positive and padding must be non-negative");
    }

    const Shape weight_shape({
        out_channels,
        in_channels,
        kernel_size,
        kernel_size,
    });

    const float fan_in =
        static_cast<float>(in_channels * kernel_size * kernel_size);
    const float fan_out =
        static_cast<float>(out_channels * kernel_size * kernel_size);

    float limit = 0.0f;
    switch (init) {
        case Conv2DInit::XavierUniform:
            limit = std::sqrt(6.0f / (fan_in + fan_out));
            break;
        case Conv2DInit::KaimingUniform:
            limit = std::sqrt(6.0f / fan_in);
            break;
    }

    weight_ = Tensor::uniform(
        weight_shape,
        DType::Float32,
        device,
        -limit,
        limit,
        next_conv2d_seed());

    bias_ = Tensor::zeros(
        Shape({out_channels}),
        DType::Float32,
        device);

    weight_.set_requires_grad(true);
    bias_.set_requires_grad(true);
}

Tensor Conv2D::forward(const Tensor& input) {
    return dl::ops::conv2d(input, weight_, bias_, stride_, padding_);
}

std::vector<Tensor*> Conv2D::parameters() {
    return {&weight_, &bias_};
}

}  // namespace dl::nn
