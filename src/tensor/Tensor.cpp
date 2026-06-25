#include "Tensor.hpp"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <utility>

namespace {
void check_cuda(cudaError_t status, const char* action) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(action) + ": " + cudaGetErrorString(status));
    }
}
}

Tensor::Tensor(const Shape& shape, DType dtype, Device device)
    : shape_(shape), dtype_(dtype), device_(device) {
    allocate();
}

Tensor::~Tensor() {
    release();
}

Tensor::Tensor(Tensor&& other) noexcept
    : data_(other.data_),
      shape_(std::move(other.shape_)),
      dtype_(other.dtype_),
      device_(other.device_) {
    other.data_ = nullptr;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        release();
        data_ = other.data_;
        shape_ = std::move(other.shape_);
        dtype_ = other.dtype_;
        device_ = other.device_;
        other.data_ = nullptr;
    }
    return *this;
}

void* Tensor::data() {
    return data_;
}

const void* Tensor::data() const {
    return data_;
}

const Shape& Tensor::shape() const {
    return shape_;
}

DType Tensor::dtype() const {
    return dtype_;
}

Device Tensor::device() const {
    return device_;
}

int64_t Tensor::numel() const {
    return shape_.numel();
}

std::size_t Tensor::nbytes() const {
    return static_cast<std::size_t>(numel()) * dtype_size(dtype_);
}

bool Tensor::defined() const {
    return data_ != nullptr;
}

void Tensor::allocate() {
    if (nbytes() == 0) {
        return;
    }
    if (!device_.is_cuda()) {
        throw std::runtime_error("Tensor currently supports CUDA allocation only");
    }
    check_cuda(cudaSetDevice(device_.index), "cudaSetDevice failed");
    check_cuda(cudaMalloc(&data_, nbytes()), "cudaMalloc failed");
}

void Tensor::release() {
    if (data_ == nullptr) {
        return;
    }
    if (device_.is_cuda()) {
        cudaSetDevice(device_.index);
        cudaFree(data_);
    }
    data_ = nullptr;
}

void Tensor::copy_from_host(const void* src, std::size_t bytes) {
    if (bytes > nbytes()) {
        throw std::runtime_error("copy_from_host exceeds tensor size");
    }
    if (bytes == 0) return;

    if (!device_.is_cuda()) {
        throw std::runtime_error("copy_from_host currently supports CUDA tensors only");
    }

    check_cuda(cudaSetDevice(device_.index), "cudaSetDevice failed");
    check_cuda(cudaMemcpy(data_, src, bytes, cudaMemcpyHostToDevice),
               "cudaMemcpy host to device failed");
}

void Tensor::copy_to_host(void* dst, std::size_t bytes) const {
    if (bytes > nbytes()) {
        throw std::runtime_error("copy_to_host exceeds tensor size");
    }
    if (bytes == 0) return;

    if (!device_.is_cuda()) {
        throw std::runtime_error("copy_to_host currently supports CUDA tensors only");
    }

    check_cuda(cudaSetDevice(device_.index), "cudaSetDevice failed");
    check_cuda(cudaMemcpy(dst, data_, bytes, cudaMemcpyDeviceToHost),
               "cudaMemcpy device to host failed");
}

void Tensor::copy_from(const Tensor& src) {
    if (src.nbytes() != nbytes()) {
        throw std::runtime_error("tensor copy size mismatch");
    }
    if (nbytes() == 0) return;

    if (src.device().is_cuda() && device_.is_cuda()) {
        check_cuda(cudaSetDevice(device_.index), "cudaSetDevice failed");
        check_cuda(cudaMemcpy(data_, src.data(), nbytes(), cudaMemcpyDeviceToDevice),
                   "cudaMemcpy device to device failed");
        return;
    }

    throw std::runtime_error("copy_from currently supports CUDA to CUDA only");
}

void Tensor::copy_to(Tensor& dst) const {
    dst.copy_from(*this);
}
