#pragma once 

#include <dl/nn/Module.hpp>
#include <dl/tensor/Tensor.hpp>

#include <memory>
#include <vector>

namespace dl::nn {

class Sequential : public Module {
public:
    void add(std::unique_ptr<Module> module);

    Tensor forward(const Tensor& input) override;

private:
    std::vector<std::unique_ptr<Module>> modules_;
};

}  // namespace dl::nn
