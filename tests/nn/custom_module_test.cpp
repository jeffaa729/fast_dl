#include <iostream>
#include <vector>

#include <dl/dl.hpp>

namespace {

class MLP : public dl::nn::Module {
public:
    explicit MLP(const dl::Device& device)
        : fc1_(3, 4, device),
          relu_(),
          fc2_(4, 2, device) {
        register_module(fc1_);
        register_module(relu_);
        register_module(fc2_);
    }

    dl::Tensor forward(const dl::Tensor& input) override {
        dl::Tensor x = fc1_(input);
        x = relu_(x);
        return fc2_(x);
    }

private:
    dl::nn::Linear fc1_;
    dl::nn::ReLU relu_;
    dl::nn::Linear fc2_;
};

}  // namespace

int main() {
    bool passed = true;

    try {
        const dl::Device device(dl::DeviceType::CUDA, 0);
        MLP model(device);

        const std::vector<float> input = {
            1.0f, 2.0f, 3.0f,
            4.0f, 5.0f, 6.0f,
        };

        dl::Tensor x = dl::Tensor::from_host<float>(
            input,
            dl::Shape({2, 3}),
            device);

        dl::Tensor y = model(x);
        const std::vector<dl::Tensor*> params = model.parameters();

        passed = y.defined() &&
                 y.shape().rank() == 2 &&
                 y.shape()[0] == 2 &&
                 y.shape()[1] == 2 &&
                 params.size() == 4 &&
                 params[0] != nullptr &&
                 params[1] != nullptr &&
                 params[2] != nullptr &&
                 params[3] != nullptr &&
                 params[0]->shape().rank() == 2 &&
                 params[0]->shape()[0] == 3 &&
                 params[0]->shape()[1] == 4 &&
                 params[1]->shape().rank() == 1 &&
                 params[1]->shape()[0] == 4 &&
                 params[2]->shape().rank() == 2 &&
                 params[2]->shape()[0] == 4 &&
                 params[2]->shape()[1] == 2 &&
                 params[3]->shape().rank() == 1 &&
                 params[3]->shape()[0] == 2;
    } catch (...) {
        passed = false;
    }

    std::cout << "custom_module_test : "
              << (passed ? "passed" : "not passed") << "\n";

    return passed ? 0 : 1;
}
