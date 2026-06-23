#pragma once
#include <cstddef>
#include <string>

enum class DType {
    Float32,
    Float16,
    Int32,
    Int64,
    Bool
};

std::size_t dtype_size(DType dtype);
std::string dtype_name(DType dtype);
