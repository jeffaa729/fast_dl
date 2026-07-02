#pragma once

#include <dl/dl.hpp>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace demo::cifar10 {

struct Cifar10Sample {
    std::vector<float> image;
    int64_t label = 0;
};

struct Cifar10Batch {
    dl::Tensor images;
    dl::Tensor labels;
    std::size_t size = 0;
};

class Cifar10Dataset : public dl::data::Dataset<Cifar10Sample> {
public:
    Cifar10Dataset(const std::string& root, bool train) {
        if (train) {
            for (int i = 1; i <= 5; ++i) {
                read_batch_file(root + "/data_batch_" + std::to_string(i) + ".bin");
            }
        } else {
            read_batch_file(root + "/test_batch.bin");
        }
    }

    std::size_t size() const override {
        return labels_.size();
    }

    Cifar10Sample get(std::size_t index) const override {
        if (index >= size()) {
            throw std::runtime_error("CIFAR-10 sample index out of range");
        }

        const std::size_t image_offset = index * image_size();
        std::vector<float> image(image_size(), 0.0f);
        for (std::size_t i = 0; i < image_size(); ++i) {
            image[i] = images_[image_offset + i];
        }

        return {std::move(image), labels_[index]};
    }

    std::size_t channels() const {
        return 3;
    }

    std::size_t rows() const {
        return 32;
    }

    std::size_t cols() const {
        return 32;
    }

    std::size_t image_size() const {
        return channels() * rows() * cols();
    }

private:
    static std::ifstream open_binary(const std::string& path) {
        std::ifstream file(path, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Failed to open CIFAR-10 file: " + path);
        }
        return file;
    }

    void read_batch_file(const std::string& path) {
        std::ifstream file = open_binary(path);

        constexpr std::size_t samples_per_file = 10000;
        constexpr std::size_t cifar_image_size = 3 * 32 * 32;
        constexpr std::size_t record_size = 1 + cifar_image_size;

        std::vector<unsigned char> record(record_size, 0);
        for (std::size_t sample = 0; sample < samples_per_file; ++sample) {
            file.read(reinterpret_cast<char*>(record.data()),
                      static_cast<std::streamsize>(record.size()));
            if (!file) {
                throw std::runtime_error("Failed to read CIFAR-10 record from: " + path);
            }

            labels_.push_back(static_cast<int64_t>(record[0]));
            const std::size_t output_base = images_.size();
            images_.resize(output_base + cifar_image_size);

            for (std::size_t channel = 0; channel < 3; ++channel) {
                for (std::size_t row = 0; row < 32; ++row) {
                    for (std::size_t col = 0; col < 32; ++col) {
                        const std::size_t src =
                            1 + channel * 1024 + row * 32 + col;
                        const std::size_t dst =
                            output_base + (channel * 32 + row) * 32 + col;
                        images_[dst] =
                            static_cast<float>(record[src]) / 255.0f;
                    }
                }
            }
        }
    }

    std::vector<float> images_;
    std::vector<int64_t> labels_;
};

inline Cifar10Batch collate_cifar10_samples(
    const std::vector<Cifar10Sample>& samples,
    const dl::Device& device) {
    if (samples.empty()) {
        throw std::runtime_error("CIFAR-10 collate requires at least one sample");
    }

    constexpr std::size_t channels = 3;
    constexpr std::size_t rows = 32;
    constexpr std::size_t cols = 32;
    constexpr std::size_t image_size = channels * rows * cols;

    const std::size_t batch_size = samples.size();
    std::vector<float> batch_images(batch_size * image_size, 0.0f);
    std::vector<int64_t> batch_labels(batch_size, 0);

    for (std::size_t sample = 0; sample < batch_size; ++sample) {
        if (samples[sample].image.size() != image_size) {
            throw std::runtime_error("CIFAR-10 samples have inconsistent image sizes");
        }

        const std::size_t offset = sample * image_size;
        for (std::size_t i = 0; i < image_size; ++i) {
            batch_images[offset + i] = samples[sample].image[i];
        }
        batch_labels[sample] = samples[sample].label;
    }

    return {
        dl::Tensor::from_host<float>(
            batch_images,
            dl::Shape({static_cast<int64_t>(batch_size),
                       static_cast<int64_t>(channels),
                       static_cast<int64_t>(rows),
                       static_cast<int64_t>(cols)}),
            device),
        dl::Tensor::from_host<int64_t>(
            batch_labels,
            dl::Shape({static_cast<int64_t>(batch_size)}),
            device),
        batch_size,
    };
}

class Cifar10DataLoader {
public:
    Cifar10DataLoader(const std::string& root,
                      bool train,
                      std::size_t batch_size,
                      const dl::Device& device,
                      bool shuffle = false,
                      bool drop_last = false,
                      uint64_t seed = 1234)
        : dataset_(root, train),
          loader_(dataset_,
                  batch_size,
                  device,
                  collate_cifar10_samples,
                  shuffle,
                  drop_last,
                  seed) {}

    Cifar10DataLoader(const Cifar10DataLoader&) = delete;
    Cifar10DataLoader& operator=(const Cifar10DataLoader&) = delete;

    std::size_t size() const {
        return dataset_.size();
    }

    std::size_t batch_size() const {
        return loader_.batch_size();
    }

    std::size_t channels() const {
        return dataset_.channels();
    }

    std::size_t rows() const {
        return dataset_.rows();
    }

    std::size_t cols() const {
        return dataset_.cols();
    }

    std::size_t image_size() const {
        return dataset_.image_size();
    }

    void reset() {
        loader_.reset();
    }

    bool has_next() const {
        return loader_.has_next();
    }

    Cifar10Batch next() {
        return loader_.next();
    }

private:
    Cifar10Dataset dataset_;
    dl::data::DataLoader<Cifar10Sample, Cifar10Batch> loader_;
};

}  // namespace demo::cifar10
