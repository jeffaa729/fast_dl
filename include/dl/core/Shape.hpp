#pragma once

#include <cstdint>
#include <initializer_list>
#include <string>
#include <vector>

class Shape {
public:
    std::vector<int64_t> dims;

    Shape() = default;
    Shape(std::initializer_list<int64_t> dims);
    explicit Shape(std::vector<int64_t> dims);

    int64_t rank() const;
    int64_t numel() const;
    int64_t operator[](int index) const;
    std::string str() const;
};
