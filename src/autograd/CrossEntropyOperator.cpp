#include <dl/autograd/CrossEntropyOperator.hpp>

#include <stdexcept>

#include <cuda_runtime.h>

#include <dl/core/CudaUtils.hpp>
#include <dl/kernels/cross_entropy.hpp>

namespace dl::autograd {

std::vector<Tensor> CrossEntropyOperator::backward(
    const std::vector<Tensor>& grad_outputs) {
    if (grad_outputs.empty()) {
        throw std::runtime_error("CrossEntropyOperator backward requires one grad output");
    }

    const std::vector<Tensor>& op_inputs = inputs();
    if (op_inputs.size() != 2) {
        throw std::runtime_error("CrossEntropyOperator backward requires two inputs");
    }

    const Tensor& logits = op_inputs[0];
    const Tensor& labels = op_inputs[1];
    const Tensor& grad_loss = grad_outputs[0];  // dL/dloss
    Tensor grad_logits(logits.shape(), logits.dtype(), logits.device());  // dL/dlogits

    dl::cuda::check(cudaSetDevice(logits.device().index), "cudaSetDevice failed");
    dl::kernels::cross_entropy_backward(
        static_cast<const float*>(logits.data()),
        static_cast<const int64_t*>(labels.data()),
        static_cast<const float*>(grad_loss.data()),
        static_cast<float*>(grad_logits.data()),
        static_cast<int>(logits.shape()[0]),
        static_cast<int>(logits.shape()[1]));
    dl::cuda::check(cudaGetLastError(), "cross_entropy backward kernel launch failed");

    return {
        grad_logits,  // dL/dlogits
        Tensor(),     // dL/dlabels, labels are integer targets
    };
}

}  // namespace dl::autograd
