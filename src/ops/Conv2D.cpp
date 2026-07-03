#include <dl/ops/Conv2D.hpp>

#include <dl/autograd/Conv2DOperator.hpp>
#include <dl/autograd/GradMode.hpp>
#include <dl/core/CudaUtils.hpp>
#include <dl/kernels/conv2d.hpp>
#include <dl/ops/OpUtils.hpp>

#include <cuda_runtime.h>

#include <memory>
#include <stdexcept>

namespace dl::ops {
    Tensor conv2d(const Tensor& input,
                  const Tensor& weight,
                  const Tensor& bias,
                  int stride,
                  int padding) {
        check_defined(input, "conv2d", "input");
        check_defined(weight, "conv2d", "weight");
        check_defined(bias, "conv2d", "bias");
        check_rank(input, 4, "conv2d", "input");
        check_rank(weight, 4, "conv2d", "weight");
        check_rank(bias, 1, "conv2d", "bias");
        check_float32(input, "conv2d", "input");    
        check_float32(weight, "conv2d", "weight");
        check_float32(bias, "conv2d", "bias");
        check_cuda(input, "conv2d", "input");
        check_cuda(weight, "conv2d", "weight");
        check_cuda(bias, "conv2d", "bias");
        check_same_device(input, weight, "conv2d", "input", "weight");
        check_same_device(input, bias, "conv2d", "input", "bias");
        
        int N = static_cast<int>(input.shape()[0]);
        int C_in = static_cast<int>(input.shape()[1]);
        int H = static_cast<int>(input.shape()[2]);
        int W = static_cast<int>(input.shape()[3]);
        int C_out = static_cast<int>(weight.shape()[0]);
        int weight_C_in = static_cast<int>(weight.shape()[1]);
        int K_h = static_cast<int>(weight.shape()[2]);
        int K_w = static_cast<int>(weight.shape()[3]);
        if (C_in != weight_C_in) {
            throw std::runtime_error("conv2d: input channels must match weight channels");
        }
        if (bias.shape()[0] != C_out) {
            throw std::runtime_error("conv2d: bias shape must match output channels");
        }
        if (stride <= 0 || padding < 0) {
            throw std::runtime_error("conv2d: stride must be positive and padding must be non-negative");
        }
        int H_out = (H + 2 * padding - K_h) / stride + 1;
        int W_out = (W + 2 * padding - K_w) / stride + 1;
        if (H_out <= 0 || W_out <= 0) {
            throw std::runtime_error("conv2d: output spatial size must be positive");
        }
        Tensor output(Shape({N, C_out, H_out, W_out}), input.dtype(), input.device());  
        dl::cuda::check(cudaSetDevice(input.device().index), "cudaSetDevice failed");
        dl::kernels::conv2d(
            static_cast<const float*>(input.data()),
            static_cast<const float*>(weight.data()),
            static_cast<const float*>(bias.data()),
            static_cast<float*>(output.data()), 
            N, C_in, H, W, C_out, K_h, K_w, H_out, W_out, stride, padding);
        dl::cuda::check(cudaGetLastError(), "conv2d kernel launch failed");
        if (dl::autograd::is_grad_enabled() &&
            (input.requires_grad() || weight.requires_grad() || bias.requires_grad())) {
            auto op = std::make_shared<dl::autograd::Conv2DOperator>(
                input.shape(),
                stride,
                padding);
            op->setup_computation_graph({input, weight, bias}, {output});
            output.set_requires_grad(true);
            output.set_creator(op);
        }
        return output;
    }
}  // namespace dl::ops
