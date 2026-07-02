#include <iostream>
#include <vector>

#include "../demo/cifar10/Cifar10DataLoader.hpp"

int main() {
    bool passed = true;

    try {
        const dl::Device device(dl::DeviceType::CUDA, 0);
        demo::cifar10::Cifar10DataLoader train_loader(
            "data/cifar-10-batches-bin",
            true,
            16,
            device);
        demo::cifar10::Cifar10DataLoader test_loader(
            "data/cifar-10-batches-bin",
            false,
            16,
            device);

        demo::cifar10::Cifar10Batch batch = train_loader.next();
        const std::vector<float> images = batch.images.to_host<float>();
        const std::vector<int64_t> labels = batch.labels.to_host<int64_t>();

        passed = train_loader.size() == 50000 &&
                 test_loader.size() == 10000 &&
                 train_loader.channels() == 3 &&
                 train_loader.rows() == 32 &&
                 train_loader.cols() == 32 &&
                 train_loader.image_size() == 3 * 32 * 32 &&
                 train_loader.batch_size() == 16 &&
                 batch.size == 16 &&
                 batch.images.shape().rank() == 4 &&
                 batch.images.shape()[0] == 16 &&
                 batch.images.shape()[1] == 3 &&
                 batch.images.shape()[2] == 32 &&
                 batch.images.shape()[3] == 32 &&
                 batch.labels.shape().rank() == 1 &&
                 batch.labels.shape()[0] == 16 &&
                 images.size() == 16 * 3 * 32 * 32 &&
                 labels.size() == 16;

        for (float value : images) {
            if (value < 0.0f || value > 1.0f) {
                passed = false;
                break;
            }
        }

        for (int64_t label : labels) {
            if (label < 0 || label > 9) {
                passed = false;
                break;
            }
        }
    } catch (...) {
        passed = false;
    }

    std::cout << "cifar10_dataloader_test : "
              << (passed ? "passed" : "not passed") << "\n";

    return passed ? 0 : 1;
}
