#include <dl/autograd/Conv2DOperator.hpp>

#include <stdexcept>

#include <cuda_runtime.h>

#include <dl/core/CudaUtils.hpp>
#include <dl/kernels/conv2d.hpp>

namespace dl::autograd {

Conv2DOperator::Conv2DOperator(const Shape& input_shape,
                               int stride,
                               int padding)
    : input_shape_(input_shape),
      stride_(stride),
      padding_(padding) {}

std::vector<Tensor> Conv2DOperator::backward(
    const std::vector<Tensor>& grad_outputs) {
    if (grad_outputs.empty()) {
        throw std::runtime_error("Conv2DOperator backward requires one grad output");
    }

    const std::vector<Tensor>& op_inputs = inputs();
    if (op_inputs.size() != 3) {
        throw std::runtime_error("Conv2DOperator backward requires three inputs");
    }

    const Tensor& input = op_inputs[0];
    const Tensor& weight = op_inputs[1];
    const Tensor& bias = op_inputs[2];
    const Tensor& grad_output = grad_outputs[0];  // dL/doutput

    const int batch_size = static_cast<int>(input.shape()[0]);
    const int C_in = static_cast<int>(input.shape()[1]);
    const int H = static_cast<int>(input.shape()[2]);
    const int W = static_cast<int>(input.shape()[3]);
    const int C_out = static_cast<int>(weight.shape()[0]);
    const int K_h = static_cast<int>(weight.shape()[2]);
    const int K_w = static_cast<int>(weight.shape()[3]);
    const int H_out = static_cast<int>(grad_output.shape()[2]);
    const int W_out = static_cast<int>(grad_output.shape()[3]);

    Tensor grad_input(input.shape(), input.dtype(), input.device());  // dL/dinput
    Tensor grad_weight(weight.shape(), weight.dtype(), weight.device());  // dL/dweight
    Tensor grad_bias(bias.shape(), bias.dtype(), bias.device());  // dL/dbias

    dl::cuda::check(cudaSetDevice(input.device().index), "cudaSetDevice failed");

    dl::kernels::conv2d_backward_input(
        static_cast<const float*>(grad_output.data()),
        static_cast<const float*>(weight.data()),
        static_cast<float*>(grad_input.data()),
        batch_size,
        C_in,
        H,
        W,
        C_out,
        K_h,
        K_w,
        H_out,
        W_out,
        stride_,
        padding_);

    dl::kernels::conv2d_backward_weight(
        static_cast<const float*>(input.data()),
        static_cast<const float*>(grad_output.data()),
        static_cast<float*>(grad_weight.data()),
        batch_size,
        C_in,
        H,
        W,
        C_out,
        K_h,
        K_w,
        H_out,
        W_out,
        stride_,
        padding_);

    dl::kernels::conv2d_backward_bias(
        static_cast<const float*>(grad_output.data()),
        static_cast<float*>(grad_bias.data()),
        batch_size,
        C_out,
        H_out,
        W_out);

    dl::cuda::check(cudaGetLastError(), "conv2d backward kernel launch failed");

    return {
        grad_input,   // dL/dinput
        grad_weight,  // dL/dweight
        grad_bias,    // dL/dbias
    };
}

}  // namespace dl::autograd
