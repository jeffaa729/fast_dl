#include <dl/autograd/MaxPool2DOperator.hpp>

#include <dl/core/CudaUtils.hpp>
#include <dl/kernels/maxpool2d.hpp>

#include <cuda_runtime.h>

#include <stdexcept>

namespace dl::autograd {

MaxPool2DOperator::MaxPool2DOperator(int kernel_size,
                                     int stride,
                                     int padding)
    : kernel_size_(kernel_size),
      stride_(stride),
      padding_(padding) {}

std::vector<Tensor> MaxPool2DOperator::backward(
    const std::vector<Tensor>& grad_outputs) {
    if (grad_outputs.empty()) {
        throw std::runtime_error("MaxPool2DOperator backward requires one grad output");
    }

    const std::vector<Tensor>& op_inputs = inputs();
    if (op_inputs.size() != 1) {
        throw std::runtime_error("MaxPool2DOperator backward requires one input");
    }

    const Tensor& input = op_inputs[0];
    const Tensor& grad_output = grad_outputs[0];  // dL/doutput

    const int N = static_cast<int>(input.shape()[0]);
    const int C = static_cast<int>(input.shape()[1]);
    const int H = static_cast<int>(input.shape()[2]);
    const int W = static_cast<int>(input.shape()[3]);
    const int H_out = static_cast<int>(grad_output.shape()[2]);
    const int W_out = static_cast<int>(grad_output.shape()[3]);

    Tensor grad_input = Tensor::zeros(input.shape(), input.dtype(), input.device());  // dL/dinput

    dl::cuda::check(cudaSetDevice(input.device().index), "cudaSetDevice failed");
    dl::kernels::max_pool2d_backward(
        static_cast<const float*>(input.data()),
        static_cast<const float*>(grad_output.data()),
        static_cast<float*>(grad_input.data()),
        N,
        C,
        H,
        W,
        H_out,
        W_out,
        kernel_size_,
        stride_,
        padding_);
    dl::cuda::check(cudaGetLastError(), "max_pool2d backward kernel launch failed");

    return {
        grad_input,  // dL/dinput
    };
}

}  // namespace dl::autograd
