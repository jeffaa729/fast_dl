#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include <dl/core/Device.hpp>
#include <dl/core/DType.hpp>
#include <dl/core/Shape.hpp>

class TensorImpl;

class Tensor {
public:
    Tensor() = default;
    Tensor(const Shape& shape, DType dtype = DType::Float32, Device device = Device());

    ~Tensor() = default;

    Tensor(const Tensor&) = default;
    Tensor& operator=(const Tensor&) = default;

    Tensor(Tensor&& other) noexcept = default;
    Tensor& operator=(Tensor&& other) noexcept = default;

    void* data();
    const void* data() const;

    const Shape& shape() const;
    DType dtype() const;
    Device device() const;

    int64_t numel() const;
    std::size_t nbytes() const;
    bool defined() const;

    void copy_from_host(const void* src, std::size_t bytes);
    void copy_to_host(void* dst, std::size_t bytes) const;
    void copy_from(const Tensor& src);
    void copy_to(Tensor& dst) const;

private:
    std::shared_ptr<TensorImpl> impl_;
};
