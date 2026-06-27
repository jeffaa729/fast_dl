#include <cmath>
#include <iostream>
#include <vector>

#include <dl/dl.hpp>

namespace {
bool close_enough(float a, float b) {
    return std::fabs(a - b) < 1e-5f;
}

bool check_vector(const std::vector<float>& actual,
                  const std::vector<float>& expected) {
    if (actual.size() != expected.size()) {
        return false;
    }

    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (!close_enough(actual[i], expected[i])) {
            return false;
        }
    }
    return true;
}

std::vector<float> relu_cpu(const std::vector<float>& input) {
    std::vector<float> output(input.size());
    for (std::size_t i = 0; i < input.size(); ++i) {
        output[i] = std::max(0.0f, input[i]);
    }
    return output;
}

std::vector<float> leaky_relu_cpu(const std::vector<float>& input, float alpha) {
    std::vector<float> output(input.size());
    for (std::size_t i = 0; i < input.size(); ++i) {
        output[i] = (input[i] > 0.0f) ? input[i] : alpha * input[i];
    }
    return output;
}

std::vector<float> gelu_cpu(const std::vector<float>& input) {
    std::vector<float> output(input.size());
    for (std::size_t i = 0; i < input.size(); ++i) {
        float x = input[i];
        output[i] = 0.5f * x * (1.0f + std::tanh(0.79788456f * (x + 0.044715f * x * x * x)));
    }
    return output;
}

std::vector<float> sigmoid_cpu(const std::vector<float>& input) {
    std::vector<float> output(input.size());
    for (std::size_t i = 0; i < input.size(); ++i) {
        output[i] = 1.0f / (1.0f + std::exp(-input[i]));
    }
    return output;
}

std::vector<float> tanh_cpu(const std::vector<float>& input) {
    std::vector<float> output(input.size());
    for (std::size_t i = 0; i < input.size(); ++i) {
        output[i] = std::tanh(input[i]);
    }
    return output;
}

}  // namespace


int main() {
    const std::vector<float> input = {
        -2.0f, -1.0f, 0.0f,
        1.0f, 2.0f, 3.0f,
    };

    dl::Tensor t_input = dl::Tensor::from_host<float>(
        input,
        dl::Shape({2, 3}),
        dl::Device(dl::DeviceType::CUDA, 0));

    const std::vector<float> expected_relu = relu_cpu(input);
    const std::vector<float> expected_leaky_relu = leaky_relu_cpu(input, 0.01f);
    const std::vector<float> expected_gelu = gelu_cpu(input);
    const std::vector<float> expected_sigmoid = sigmoid_cpu(input);
    const std::vector<float> expected_tanh = tanh_cpu(input);

    bool passed = true;
    passed = passed && check_vector(dl::ops::relu(t_input).to_host<float>(), expected_relu);
    passed = passed && check_vector(dl::ops::leaky_relu(t_input, 0.01f).to_host<float>(), expected_leaky_relu);
    passed = passed && check_vector(dl::ops::gelu(t_input).to_host<float>(), expected_gelu);
    passed = passed && check_vector(dl::ops::sigmoid(t_input).to_host<float>(), expected_sigmoid);
    passed = passed && check_vector(dl::ops::tanh(t_input).to_host<float>(), expected_tanh);

    std::cout << "activation tensor test: " << (passed ? "passed" : "not passed") << std::endl;
    return passed ? 0 : 1;
}