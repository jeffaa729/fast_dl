#include <dl/ops/Conv2D.hpp>

#include <cuda_runtime.h>
#include <dl/ops/OpUtils.hpp>
//#include <dl/autograd/Conv2DOperator.hpp>
#include <dl/autograd/GradMode.hpp>
#include <dl/core/CudaUtils.hpp>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <conv2d.hpp>

namespace dl::ops {
    Tensor conv2d(const Tensor& input, const Tensor& weight, const Tensor& bias,
                  int stride, int padding) {
        check_defined(input, "conv2d", "input");
        check_defined(weight, "conv2d", "weight");
        check_defined(bias, "conv2d", "bias");
        check_float32(input, "conv2d", "input");
        check_float32(weight, "conv2d", "weight");
        check_float32(bias, "conv2d", "bias");
        check_cuda(input, "conv2d", "input");
        check_cuda(weight, "conv2d", "weight");
        check_cuda(bias, "conv2d", "bias");

        if (input.shape().rank() != 4) {
            throw std::runtime_error("conv2d input tensor must have rank 4");
        }
        if (weight.shape().rank() != 4) {
            throw std::runtime_error("conv2d weight tensor must have rank 4");
        }
        if (bias.shape().rank() != 1) {
            throw std::runtime_error("conv2d bias tensor must have rank 1");
        }   
        Tensor output(Shape({input.shape()[0], weight.shape()[0],
                             input.shape()[2], input.shape()[3]}),
                      input.dtype(), input.device());
        dl::cuda::check(cudaSetDevice(input.device().index), "cudaSetDevice failed");
        dl::kernels::conv2d(
            static_cast<const float*>(input.data()),
            static_cast<const float*>(weight.data()),
            static_cast<const float*>(bias.data()),
            static_cast<float*>(output.data()),
            static_cast<int>(input.shape()[0]),
            static_cast<int>(input.shape()[1]),
            static_cast<int>(input.shape()[2]),
            static_cast<int>(input.shape()[3]),
            static_cast<int>(weight.shape()[0]),
            static_cast<int>(weight.shape()[2]),
            static_cast<int>(weight.shape()[3]),
            stride,
            padding);
        dl::cuda::check(cudaGetLastError(), "conv2d kernel launch failed");
        return output;
    }
}