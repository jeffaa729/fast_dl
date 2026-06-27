#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

#include <dl/core/Device.hpp>
#include <dl/core/DType.hpp>
#include <dl/core/Shape.hpp>

namespace dl {

class TensorImpl;

template <typename T>
DType tensor_dtype();

template <>
inline DType tensor_dtype<float>() {
    return DType::Float32;
}

template <>
inline DType tensor_dtype<int32_t>() {
    return DType::Int32;
}

template <>
inline DType tensor_dtype<int64_t>() {
    return DType::Int64;
}

class Tensor {
public:
    Tensor() = default;
    Tensor(const Shape& shape, DType dtype = DType::Float32, Device device = Device());

    static Tensor empty(const Shape& shape,
                        DType dtype = DType::Float32,
                        Device device = Device());
    static Tensor zeros(const Shape& shape,
                        DType dtype = DType::Float32,
                        Device device = Device());
    static Tensor randn(Shape shape,
                        DType dtype,
                        Device device,
                        float mean,
                        float stddev,
                        uint64_t seed = 1234);
    static Tensor uniform(Shape shape, DType dtype, Device device, float low, float high, uint64_t seed = 1234);
    static Tensor xavier_uniform(Shape shape, DType dtype, Device device, uint64_t seed = 1234);
    static Tensor kaiming_uniform(Shape shape, DType dtype, Device device, uint64_t seed = 1234);

    static Tensor empty_like(const Tensor& other);
    static Tensor zeros_like(const Tensor& other);


    template <typename T>
    static Tensor from_host(const std::vector<T>& data,
                            const Shape& shape,
                            Device device = Device());

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
    void zero_();

    template <typename T>
    std::vector<T> to_host() const;

private:
    std::shared_ptr<TensorImpl> impl_;
};

template <typename T>
Tensor Tensor::from_host(const std::vector<T>& data,
                         const Shape& shape,
                         Device device) {
    if (shape.numel() != static_cast<int64_t>(data.size())) {
        throw std::runtime_error("from_host data size does not match shape");
    }

    Tensor tensor(shape, tensor_dtype<T>(), device);
    tensor.copy_from_host(data.data(), data.size() * sizeof(T));
    return tensor;
}

template <typename T>
std::vector<T> Tensor::to_host() const {
    if (dtype() != tensor_dtype<T>()) {
        throw std::runtime_error("to_host dtype does not match requested host type");
    }

    std::vector<T> output(static_cast<std::size_t>(numel()));
    copy_to_host(output.data(), output.size() * sizeof(T));
    return output;
}

}  // namespace dl

using Tensor = dl::Tensor;
using dl::tensor_dtype;
