#include <dl/ops/OpUtils.hpp>

#include <stdexcept>
#include <string>

namespace dl::ops {

namespace {

std::string prefix(const char* op_name) {
    return std::string(op_name) + ": ";
}

std::string dtype_to_string(DType dtype) {
    return dtype_name(dtype);
}

}  // namespace

void check_defined(const Tensor& tensor, const char* op_name,
                   const char* tensor_name) {
    if (!tensor.defined()) {
        throw std::runtime_error(prefix(op_name) + tensor_name +
                                 " tensor is not defined");
    }
}

void check_rank(const Tensor& tensor, int64_t rank, const char* op_name,
                const char* tensor_name) {
    if (tensor.shape().rank() != rank) {
        throw std::runtime_error(prefix(op_name) + tensor_name +
                                 " tensor has incorrect rank");
    }
}

void check_cuda(const Tensor& tensor, const char* op_name,
                const char* tensor_name) {
    if (!tensor.device().is_cuda()) {
        throw std::runtime_error(prefix(op_name) + tensor_name +
                                 " tensor must be CUDA");
    }
}

void check_dtype(const Tensor& tensor, DType dtype, const char* op_name,
                 const char* tensor_name) {
    if (tensor.dtype() != dtype) {
        throw std::runtime_error(prefix(op_name) + tensor_name +
                                 " tensor must be " + dtype_to_string(dtype));
    }
}

void check_float32(const Tensor& tensor, const char* op_name,
                   const char* tensor_name) {
    check_dtype(tensor, DType::Float32, op_name, tensor_name);
}

void check_same_device(const Tensor& a, const Tensor& b, const char* op_name,
                       const char* a_name, const char* b_name) {
    if (a.device().type != b.device().type ||
        a.device().index != b.device().index) {
        throw std::runtime_error(prefix(op_name) + a_name + " and " +
                                 b_name + " tensors must be on the same device");
    }
}

void check_same_dtype(const Tensor& a, const Tensor& b, const char* op_name,
                      const char* a_name, const char* b_name) {
    if (a.dtype() != b.dtype()) {
        throw std::runtime_error(prefix(op_name) + a_name + " and " +
                                 b_name + " tensors must have the same dtype");
    }
}

void check_same_shape(const Tensor& a, const Tensor& b, const char* op_name,
                      const char* a_name, const char* b_name) {
    if (a.shape().rank() != b.shape().rank()) {
        throw std::runtime_error(prefix(op_name) + a_name + " and " +
                                 b_name + " tensors must have the same shape");
    }

    for (int64_t dim = 0; dim < a.shape().rank(); ++dim) {
        if (a.shape()[static_cast<int>(dim)] !=
            b.shape()[static_cast<int>(dim)]) {
            throw std::runtime_error(prefix(op_name) + a_name + " and " +
                                     b_name +
                                     " tensors must have the same shape");
        }
    }
}

void check_binary_float_cuda_op(const Tensor& a, const Tensor& b,
                                const char* op_name) {
    check_defined(a, op_name, "left");
    check_defined(b, op_name, "right");
    check_same_shape(a, b, op_name);
    check_same_dtype(a, b, op_name);
    check_float32(a, op_name, "left");
    check_cuda(a, op_name, "left");
    check_cuda(b, op_name, "right");
    check_same_device(a, b, op_name);
}

}  // namespace dl::ops
