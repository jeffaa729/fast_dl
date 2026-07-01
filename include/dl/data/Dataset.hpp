#pragma once

#include <cstddef>

namespace dl::data {

template <typename Sample>
class Dataset {
public:
    using SampleType = Sample;

    virtual ~Dataset() = default;

    virtual std::size_t size() const = 0;
    virtual Sample get(std::size_t index) const = 0;
};

}  // namespace dl::data
