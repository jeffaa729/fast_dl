#include <cmath>
#include <iostream>
#include <vector>

#include <dl/dl.hpp>

int main() {
    bool passed = true;

    try {
        constexpr int in_features = 4;
        constexpr int out_features = 3;
        const dl::Device device(dl::DeviceType::CUDA, 0);

        dl::nn::Linear layer(
            in_features,
            out_features,
            device,
            dl::nn::LinearInit::XavierUniform);
        const std::vector<dl::Tensor*> params = layer.parameters();

        passed = params.size() == 2 &&
                 params[0] != nullptr &&
                 params[1] != nullptr &&
                 params[0]->shape().rank() == 2 &&
                 params[0]->shape()[0] == in_features &&
                 params[0]->shape()[1] == out_features &&
                 params[1]->shape().rank() == 1 &&
                 params[1]->shape()[0] == out_features;

        const float limit = std::sqrt(
            6.0f / static_cast<float>(in_features + out_features));
        const std::vector<float> weight = params[0]->to_host<float>();
        const std::vector<float> bias = params[1]->to_host<float>();

        bool weight_has_nonzero = false;
        for (float value : weight) {
            if (value < -limit || value > limit) {
                passed = false;
                break;
            }
            if (std::fabs(value) > 1e-6f) {
                weight_has_nonzero = true;
            }
        }

        bool bias_all_zero = true;
        for (float value : bias) {
            if (std::fabs(value) > 1e-6f) {
                bias_all_zero = false;
                break;
            }
        }

        passed = passed && weight_has_nonzero && bias_all_zero;
    } catch (...) {
        passed = false;
    }

    std::cout << "linear_init_test : "
              << (passed ? "passed" : "not passed") << "\n";

    return passed ? 0 : 1;
}
