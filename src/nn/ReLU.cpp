#include <dl/nn/ReLU.hpp>

#include <dl/ops/Activation.hpp>

namespace dl::nn {

Tensor ReLU::forward(const Tensor& input) {
    return dl::ops::relu(input);
}

}  // namespace dl::nn
