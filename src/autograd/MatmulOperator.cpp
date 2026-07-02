#include <dl/autograd/MatmulOperator.hpp>

#include <stdexcept>

#include <cuda_runtime.h>

#include <dl/core/CudaUtils.hpp>
#include <dl/kernels/gemm.hpp>

namespace dl::autograd {

std::vector<Tensor> MatmulOperator::backward(
    const std::vector<Tensor>& grad_outputs) {
    if (grad_outputs.empty()) {
        throw std::runtime_error("MatmulOperator backward requires one grad output");
    }

    const std::vector<Tensor>& op_inputs = inputs();
    if (op_inputs.size() != 2) {
        throw std::runtime_error("MatmulOperator backward requires two inputs");
    }

    const Tensor& a = op_inputs[0];
    const Tensor& b = op_inputs[1];
    const Tensor& grad_output = grad_outputs[0];  // dL/doutput

    const int m = static_cast<int>(a.shape()[0]);
    const int k = static_cast<int>(a.shape()[1]);
    const int n = static_cast<int>(b.shape()[1]);

    Tensor grad_a(Shape({a.shape()[0], a.shape()[1]}), a.dtype(), a.device());  // dL/da
    Tensor grad_b(Shape({b.shape()[0], b.shape()[1]}), b.dtype(), b.device());  // dL/db

    dl::cuda::check(cudaSetDevice(a.device().index), "cudaSetDevice failed");

    dl::kernels::gemm(
        static_cast<const float*>(grad_output.data()),
        static_cast<const float*>(b.data()),
        static_cast<float*>(grad_a.data()),
        m,
        k,
        n,
        false,
        true);

    dl::kernels::gemm(
        static_cast<const float*>(a.data()),
        static_cast<const float*>(grad_output.data()),
        static_cast<float*>(grad_b.data()),
        k,
        n,
        m,
        true,
        false);

    dl::cuda::check(cudaGetLastError(), "matmul backward kernel launch failed");

    return {
        grad_a,  // dL/da
        grad_b,  // dL/db
    };
}

}  // namespace dl::autograd
