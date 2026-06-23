#pragma once

#include <cstddef>
#include <cstdint>

#include "Device.hpp"
#include "DType.hpp"
#include "Shape.hpp"

class Tensor {
public:
    Tensor() = default;
    Tensor(const Shape& shape, DType dtype = DType::Float32, Device device = Device());

    ~Tensor();

    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;

    void* data();
    const void* data() const;

    const Shape& shape() const;
    DType dtype() const;
    Device device() const;

    int64_t numel() const;
    std::size_t nbytes() const;
    bool defined() const;

private:
    void* data_ = nullptr;
    Shape shape_;
    DType dtype_ = DType::Float32;
    Device device_;

    void allocate();
    void release();
};
