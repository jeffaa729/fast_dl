#include <dl/core/Device.hpp>

namespace dl {

Device::Device(DeviceType type, int index) : type(type), index(index) {}

bool Device::is_cuda() const {
    return type == DeviceType::CUDA;
}

bool Device::is_cpu() const {
    return type == DeviceType::CPU;
}

std::string Device::str() const {
    if (is_cuda()) {
        return "cuda:" + std::to_string(index);
    }
    return "cpu";
}

}  // namespace dl
