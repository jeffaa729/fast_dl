#pragma once

#include <string>

namespace dl {

enum class DeviceType {
    CPU,
    CUDA
};

class Device {
public:
    DeviceType type;
    int index;

    Device(DeviceType type = DeviceType::CUDA, int index = 0);

    bool is_cuda() const;
    bool is_cpu() const;
    std::string str() const;
};

}  // namespace dl

using DeviceType = dl::DeviceType;
using Device = dl::Device;
