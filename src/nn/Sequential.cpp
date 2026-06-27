#include <dl/nn/Sequential.hpp>

#include <stdexcept>
#include <utility>

namespace dl::nn {

void Sequential::add(std::unique_ptr<Module> module) {
    if (!module) {
        throw std::runtime_error("Sequential cannot add a null module");
    }
    modules_.push_back(std::move(module));
    register_module(*modules_.back());
}

Tensor Sequential::forward(const Tensor& input) {
    Tensor output = input;
    for (const auto& module : modules_) {
        output = module->forward(output);
    }
    return output;
}

}  // namespace dl::nn
