#pragma once

#include <dl/tensor/Tensor.hpp>

#include <vector>

namespace dl::nn {

class Module {
public:
    virtual ~Module() = default;

    virtual Tensor forward(const Tensor& input) = 0;

    virtual std::vector<Tensor*> parameters() {
        std::vector<Tensor*> result;
        for (Module* child : children_) {
            std::vector<Tensor*> child_params = child->parameters();
            result.insert(result.end(), child_params.begin(), child_params.end());
        }
        return result;
    }

    Tensor operator()(const Tensor& input) {
        return forward(input);
    }

    void train() {
        training_ = true;
        for (Module* child : children_) {
            child->train();
        }
    }

    void eval() {
        training_ = false;
        for (Module* child : children_) {
            child->eval();
        }
    }

    bool is_training() const {
        return training_;
    }

protected:
    void register_module(Module& module) {
        children_.push_back(&module);
    }

private:
    bool training_ = true;
    std::vector<Module*> children_;
};

}  // namespace dl::nn
