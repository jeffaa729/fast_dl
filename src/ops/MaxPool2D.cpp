#include <dl/ops/MaxPool2D.hpp>

#include <dl/autograd/GradMode.hpp>
#include <dl/autograd/MaxPool2DOperator.hpp>
#include <dl/core/CudaUtils.hpp>
#include <dl/kernels/maxpool2d.hpp>
#include <dl/ops/OpUtils.hpp>

#include <cuda_runtime.h>

#include <memory>
#include <stdexcept>

namespace dl::ops {

Tensor max_pool2d(const Tensor& input,
                  int kernel_size,
                  int stride,
                  int padding) {
    check_defined(input, "max_pool2d", "input");
    check_rank(input, 4, "max_pool2d", "input");
    check_float32(input, "max_pool2d", "input");
    check_cuda(input, "max_pool2d", "input");

    if (kernel_size <= 0) {
        throw std::runtime_error("max_pool2d: kernel size must be positive");
    }
    if (stride == -1) {
        stride = kernel_size;
    }
    if (stride <= 0 || padding < 0) {
        throw std::runtime_error("max_pool2d: stride must be positive and padding must be non-negative");
    }

    const int N = static_cast<int>(input.shape()[0]);
    const int C = static_cast<int>(input.shape()[1]);
    const int H = static_cast<int>(input.shape()[2]);
    const int W = static_cast<int>(input.shape()[3]);
    const int H_out = (H + 2 * padding - kernel_size) / stride + 1;
    const int W_out = (W + 2 * padding - kernel_size) / stride + 1;
    if (H_out <= 0 || W_out <= 0) {
        throw std::runtime_error("max_pool2d: output spatial size must be positive");
    }

    Tensor output(Shape({N, C, H_out, W_out}), input.dtype(), input.device());

    dl::cuda::check(cudaSetDevice(input.device().index), "cudaSetDevice failed");
    dl::kernels::max_pool2d_forward(
        static_cast<const float*>(input.data()),
        static_cast<float*>(output.data()),
        N,
        C,
        H,
        W,
        H_out,
        W_out,
        kernel_size,
        stride,
        padding);
    dl::cuda::check(cudaGetLastError(), "max_pool2d kernel launch failed");

    if (dl::autograd::is_grad_enabled() && input.requires_grad()) {
        auto op = std::make_shared<dl::autograd::MaxPool2DOperator>(
            kernel_size,
            stride,
            padding);
        op->setup_computation_graph({input}, {output});
        output.set_requires_grad(true);
        output.set_creator(op);
    }

    return output;
}

}  // namespace dl::ops
