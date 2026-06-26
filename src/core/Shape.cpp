#include <dl/core/Shape.hpp>

#include <numeric>
#include <sstream>
#include <utility>

namespace dl {

Shape::Shape(std::initializer_list<int64_t> dims) : dims(dims) {}

Shape::Shape(std::vector<int64_t> dims) : dims(std::move(dims)) {}

int64_t Shape::rank() const {
    return static_cast<int64_t>(dims.size());
}

int64_t Shape::numel() const {
    if (dims.empty()) {
        return 0;
    }
    return std::accumulate(dims.begin(), dims.end(), int64_t{1},
                           [](int64_t a, int64_t b) { return a * b; });
}

int64_t Shape::operator[](int index) const {
    return dims[index];
}

std::string Shape::str() const {
    std::ostringstream out;
    out << "(";
    for (std::size_t i = 0; i < dims.size(); ++i) {
        if (i > 0) {
            out << ", ";
        }
        out << dims[i];
    }
    out << ")";
    return out.str();
}

}  // namespace dl
