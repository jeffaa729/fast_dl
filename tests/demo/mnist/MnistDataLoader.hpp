#pragma once

#include <dl/dl.hpp>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace demo::mnist {

struct MnistSample {
    std::vector<float> image;
    int64_t label = 0;
};

struct MnistBatch {
    dl::Tensor images;
    dl::Tensor labels;
    std::size_t size = 0;
};

class MnistDataset : public dl::data::Dataset<MnistSample> {
public:
    MnistDataset(const std::string& image_path,
                 const std::string& label_path) {
        read_images(image_path);
        read_labels(label_path);

        if (labels_.size() != image_count_) {
            throw std::runtime_error("MNIST image and label counts do not match");
        }
    }

    std::size_t size() const override {
        return image_count_;
    }

    MnistSample get(std::size_t index) const override {
        if (index >= image_count_) {
            throw std::runtime_error("MNIST sample index out of range");
        }

        const std::size_t pixels = image_size();
        const std::size_t offset = index * pixels;
        std::vector<float> image(pixels, 0.0f);

        for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
            image[pixel] =
                static_cast<float>(images_[offset + pixel]) / 255.0f;
        }

        return {std::move(image), labels_[index]};
    }

    std::size_t rows() const {
        return rows_;
    }

    std::size_t cols() const {
        return cols_;
    }

    std::size_t image_size() const {
        return rows_ * cols_;
    }

private:
    static uint32_t read_u32_be(std::ifstream& file) {
        unsigned char bytes[4] = {0, 0, 0, 0};
        file.read(reinterpret_cast<char*>(bytes), sizeof(bytes));
        if (!file) {
            throw std::runtime_error("Failed to read MNIST u32 field");
        }

        return (static_cast<uint32_t>(bytes[0]) << 24) |
               (static_cast<uint32_t>(bytes[1]) << 16) |
               (static_cast<uint32_t>(bytes[2]) << 8) |
               static_cast<uint32_t>(bytes[3]);
    }

    static std::ifstream open_binary(const std::string& path) {
        std::ifstream file(path, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Failed to open MNIST file: " + path);
        }
        return file;
    }

    void read_images(const std::string& path) {
        std::ifstream file = open_binary(path);

        const uint32_t magic = read_u32_be(file);
        if (magic != 2051) {
            throw std::runtime_error("MNIST image file has wrong magic number");
        }

        image_count_ = read_u32_be(file);
        rows_ = read_u32_be(file);
        cols_ = read_u32_be(file);

        const std::size_t total_pixels = image_count_ * rows_ * cols_;
        images_.resize(total_pixels);
        file.read(reinterpret_cast<char*>(images_.data()),
                  static_cast<std::streamsize>(images_.size()));
        if (!file) {
            throw std::runtime_error("Failed to read MNIST image data");
        }
    }

    void read_labels(const std::string& path) {
        std::ifstream file = open_binary(path);

        const uint32_t magic = read_u32_be(file);
        if (magic != 2049) {
            throw std::runtime_error("MNIST label file has wrong magic number");
        }

        const std::size_t label_count = read_u32_be(file);
        labels_.resize(label_count);

        for (std::size_t i = 0; i < label_count; ++i) {
            unsigned char label = 0;
            file.read(reinterpret_cast<char*>(&label), sizeof(label));
            if (!file) {
                throw std::runtime_error("Failed to read MNIST label data");
            }
            labels_[i] = static_cast<int64_t>(label);
        }
    }

    std::size_t image_count_ = 0;
    std::size_t rows_ = 0;
    std::size_t cols_ = 0;
    std::vector<uint8_t> images_;
    std::vector<int64_t> labels_;
};

inline MnistBatch collate_mnist_samples(
    const std::vector<MnistSample>& samples,
    const dl::Device& device) {
    if (samples.empty()) {
        throw std::runtime_error("MNIST collate requires at least one sample");
    }

    const std::size_t batch_size = samples.size();
    const std::size_t pixels = samples[0].image.size();
    std::vector<float> batch_images(batch_size * pixels, 0.0f);
    std::vector<int64_t> batch_labels(batch_size, 0);

    for (std::size_t row = 0; row < batch_size; ++row) {
        if (samples[row].image.size() != pixels) {
            throw std::runtime_error("MNIST samples have inconsistent image sizes");
        }

        const std::size_t offset = row * pixels;
        for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
            batch_images[offset + pixel] = samples[row].image[pixel];
        }
        batch_labels[row] = samples[row].label;
    }

    return {
        dl::Tensor::from_host<float>(
            batch_images,
            dl::Shape({static_cast<int64_t>(batch_size),
                       static_cast<int64_t>(pixels)}),
            device),
        dl::Tensor::from_host<int64_t>(
            batch_labels,
            dl::Shape({static_cast<int64_t>(batch_size)}),
            device),
        batch_size,
    };
}

class MnistDataLoader {
public:
    MnistDataLoader(const std::string& image_path,
                    const std::string& label_path,
                    std::size_t batch_size,
                    const dl::Device& device)
        : dataset_(image_path, label_path),
          loader_(dataset_, batch_size, device, collate_mnist_samples) {}

    MnistDataLoader(const MnistDataLoader&) = delete;
    MnistDataLoader& operator=(const MnistDataLoader&) = delete;

    std::size_t size() const {
        return dataset_.size();
    }

    std::size_t batch_size() const {
        return loader_.batch_size();
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

    MnistBatch next() {
        return loader_.next();
    }

private:
    MnistDataset dataset_;
    dl::data::DataLoader<MnistSample, MnistBatch> loader_;
};

}  // namespace demo::mnist
