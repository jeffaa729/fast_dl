#include <iostream>
#include <vector>

#include "mnist/MnistDataLoader.hpp"

int main() {
    bool passed = true;

    try {
        const dl::Device device(dl::DeviceType::CUDA, 0);
        demo::mnist::MnistDataLoader loader(
            "mnist/t10k-images-idx3-ubyte",
            "mnist/t10k-labels-idx1-ubyte",
            32,
            device);

        demo::mnist::MnistBatch batch = loader.next();
        const std::vector<float> images = batch.images.to_host<float>();
        const std::vector<int64_t> labels = batch.labels.to_host<int64_t>();

        passed = loader.size() == 10000 &&
                 loader.rows() == 28 &&
                 loader.cols() == 28 &&
                 loader.image_size() == 784 &&
                 loader.batch_size() == 32 &&
                 batch.size == 32 &&
                 batch.images.shape().rank() == 2 &&
                 batch.images.shape()[0] == 32 &&
                 batch.images.shape()[1] == 784 &&
                 batch.labels.shape().rank() == 1 &&
                 batch.labels.shape()[0] == 32 &&
                 images.size() == 32 * 784 &&
                 labels.size() == 32;

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

    std::cout << "mnist_dataloader_demo : "
              << (passed ? "passed" : "not passed") << "\n";

    return passed ? 0 : 1;
}
