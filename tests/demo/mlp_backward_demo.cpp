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
        return fc2_(relu_(fc1_(input)));
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

        dl::Tensor input = dl::Tensor::from_host<float>(
            {
                1.0f, 2.0f, 3.0f,
                4.0f, 5.0f, 6.0f,
            },
            dl::Shape({2, 3}),
            device);
        input.set_requires_grad(true);

        dl::Tensor output = model(input);
        output.backward();

        const std::vector<dl::Tensor*> params = model.parameters();

        passed = output.defined() &&
                 output.shape().rank() == 2 &&
                 output.shape()[0] == 2 &&
                 output.shape()[1] == 2 &&
                 input.grad().defined() &&
                 params.size() == 4;

        for (const dl::Tensor* param : params) {
            if (param == nullptr || !param->grad().defined()) {
                passed = false;
                break;
            }
        }
    } catch (...) {
        passed = false;
    }

    std::cout << "mlp_backward_demo : "
              << (passed ? "passed" : "not passed") << "\n";

    return passed ? 0 : 1;
}
