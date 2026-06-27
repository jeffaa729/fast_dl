#include <iostream>

#include <dl/dl.hpp>

namespace {

class ModeModel : public dl::nn::Module {
public:
    explicit ModeModel(const dl::Device& device)
        : fc_(2, 2, device),
          relu_() {
        register_module(fc_);
        register_module(relu_);
    }

    dl::Tensor forward(const dl::Tensor& input) override {
        return relu_(fc_(input));
    }

    bool children_are_training() const {
        return fc_.is_training() && relu_.is_training();
    }

    bool children_are_eval() const {
        return !fc_.is_training() && !relu_.is_training();
    }

private:
    dl::nn::Linear fc_;
    dl::nn::ReLU relu_;
};

}  // namespace

int main() {
    bool passed = true;

    try {
        const dl::Device device(dl::DeviceType::CUDA, 0);
        ModeModel model(device);

        passed = model.is_training() && model.children_are_training();

        model.eval();
        passed = passed && !model.is_training() && model.children_are_eval();

        model.train();
        passed = passed && model.is_training() && model.children_are_training();
    } catch (...) {
        passed = false;
    }

    std::cout << "module_mode_test : "
              << (passed ? "passed" : "not passed") << "\n";

    return passed ? 0 : 1;
}
