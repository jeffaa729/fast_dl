#include <dl/core/DType.hpp>

#include <stdexcept>

namespace dl {

std::size_t dtype_size(DType dtype) {
    switch (dtype) {
        case DType::Float32:
            return 4;
        case DType::Float16:
            return 2;
        case DType::Int32:
            return 4;
        case DType::Int64:
            return 8;
        case DType::Bool:
            return 1;
    }
    throw std::invalid_argument("unknown dtype");
}

std::string dtype_name(DType dtype) {
    switch (dtype) {
        case DType::Float32:
            return "float32";
        case DType::Float16:
            return "float16";
        case DType::Int32:
            return "int32";
        case DType::Int64:
            return "int64";
        case DType::Bool:
            return "bool";
    }
    throw std::invalid_argument("unknown dtype");
}

}  // namespace dl
