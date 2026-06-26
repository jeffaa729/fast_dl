#include <iostream>
#include <vector>

#include <dl/dl.hpp>

int main() {
    dl::Tensor x = dl::Tensor::empty(
        dl::Shape({2, 3}),
        dl::DType::Float32,
        dl::Device(dl::DeviceType::CUDA, 0));
    dl::Tensor z = dl::Tensor::zeros(
        dl::Shape({2, 3}),
        dl::DType::Float32,
        dl::Device(dl::DeviceType::CUDA, 0));
    dl::Tensor x_like = dl::Tensor::empty_like(x);
    dl::Tensor z_like = dl::Tensor::zeros_like(x);
    const std::vector<float> zeros = z.to_host<float>();
    const std::vector<float> zeros_like = z_like.to_host<float>();

    bool zero_check = true;
    for (float value : zeros) {
        if (value != 0.0f) {
            zero_check = false;
            break;
        }
    }
    for (float value : zeros_like) {
        if (value != 0.0f) {
            zero_check = false;
            break;
        }
    }

    const bool passed = (x.numel() == 6 && x.nbytes() == 24 &&
                         x_like.numel() == x.numel() &&
                         x_like.dtype() == x.dtype() &&
                         x_like.device().type == x.device().type &&
                         x_like.device().index == x.device().index &&
                         zero_check);
    std::cout << "tensor_test : " << (passed ? "passed" : "not passed") << "\n";

    return passed ? 0 : 1;
}
