#pragma once

#include <cstdint>

#include <dl/tensor/Tensor.hpp>

namespace dl::ops {

void check_defined(const Tensor& tensor, const char* op_name,
                   const char* tensor_name = "input");
void check_rank(const Tensor& tensor, int64_t rank, const char* op_name,
                const char* tensor_name = "input");
void check_cuda(const Tensor& tensor, const char* op_name,
                const char* tensor_name = "input");
void check_dtype(const Tensor& tensor, DType dtype, const char* op_name,
                 const char* tensor_name = "input");
void check_float32(const Tensor& tensor, const char* op_name,
                   const char* tensor_name = "input");
void check_same_device(const Tensor& a, const Tensor& b, const char* op_name,
                       const char* a_name = "left",
                       const char* b_name = "right");
void check_same_dtype(const Tensor& a, const Tensor& b, const char* op_name,
                      const char* a_name = "left",
                      const char* b_name = "right");
void check_same_shape(const Tensor& a, const Tensor& b, const char* op_name,
                      const char* a_name = "left",
                      const char* b_name = "right");
void check_binary_float_cuda_op(const Tensor& a, const Tensor& b,
                                const char* op_name);

}  // namespace dl::ops
