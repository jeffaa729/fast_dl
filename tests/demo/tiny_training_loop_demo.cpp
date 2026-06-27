#include <iostream>
#include <vector>

#include <dl/dl.hpp>

namespace {

class TinyMLP : public dl::nn::Module {
public:
    explicit TinyMLP(const dl::Device& device)
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
        TinyMLP model(device);
        dl::optim::SGD optimizer(model.parameters(), 0.05f);

        dl::Tensor input = dl::Tensor::from_host<float>(
            {
                1.0f, 0.0f, 1.0f,
                0.0f, 1.0f, 1.0f,
            },
            dl::Shape({2, 3}),
            device);
        dl::Tensor labels = dl::Tensor::from_host<int64_t>(
            {0, 1},
            dl::Shape({2}),
            device);

        float initial_loss = 0.0f;
        float final_loss = 0.0f;
        bool saw_gradients = false;

        for (int step = 0; step < 20; ++step) {
            optimizer.zero_grad();

            dl::Tensor logits = model(input);
            dl::Tensor loss = dl::ops::cross_entropy(logits, labels);
            const float loss_value = loss.to_host<float>()[0];
            if (step == 0) {
                initial_loss = loss_value;
            }

            loss.backward();

            const std::vector<dl::Tensor*> params = model.parameters();
            passed = passed &&
                     loss.defined() &&
                     loss.shape().rank() == 1 &&
                     loss.shape()[0] == 1 &&
                     params.size() == 4;

            bool step_has_gradients = params.size() == 4;
            for (const dl::Tensor* param : params) {
                if (param == nullptr || !param->grad().defined()) {
                    step_has_gradients = false;
                    passed = false;
                    break;
                }
            }

            saw_gradients = saw_gradients || step_has_gradients;
            optimizer.step();
        }

        optimizer.zero_grad();

        dl::Tensor final_logits = model(input);
        dl::Tensor final_loss_tensor = dl::ops::cross_entropy(final_logits, labels);
        final_loss = final_loss_tensor.to_host<float>()[0];

        const std::vector<dl::Tensor*> params = model.parameters();
        for (const dl::Tensor* param : params) {
            passed = passed &&
                     param != nullptr &&
                     !param->grad().defined();
        }

        passed = passed && saw_gradients && final_loss < initial_loss;
    } catch (...) {
        passed = false;
    }

    std::cout << "tiny_training_loop_demo : "
              << (passed ? "passed" : "not passed") << "\n";

    return passed ? 0 : 1;
}
