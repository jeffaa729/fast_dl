#include <dl/nn/MaxPool2D.hpp>

#include <dl/ops/MaxPool2D.hpp>

#include <stdexcept>

namespace dl::nn {

MaxPool2D::MaxPool2D(int kernel_size, int stride, int padding)
    : kernel_size_(kernel_size),
      stride_(stride),
      padding_(padding) {
    if (kernel_size <= 0) {
        throw std::runtime_error("MaxPool2D kernel size must be positive");
    }
    if (stride != -1 && stride <= 0) {
        throw std::runtime_error("MaxPool2D stride must be positive");
    }
    if (padding < 0) {
        throw std::runtime_error("MaxPool2D padding must be non-negative");
    }
}

Tensor MaxPool2D::forward(const Tensor& input) {
    return dl::ops::max_pool2d(input, kernel_size_, stride_, padding_);
}

}  // namespace dl::nn
