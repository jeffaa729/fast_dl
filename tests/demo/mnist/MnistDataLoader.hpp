#pragma once

#include <dl/dl.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace demo::mnist {

struct MnistBatch {
    dl::Tensor images;
    dl::Tensor labels;
    std::size_t size = 0;
};

class MnistDataLoader {
public:
    MnistDataLoader(const std::string& image_path,
                    const std::string& label_path,
                    std::size_t batch_size,
                    const dl::Device& device)
        : batch_size_(batch_size),
          device_(device) {
        if (batch_size_ == 0) {
            throw std::runtime_error("MNIST batch size must be positive");
        }

        read_images(image_path);
        read_labels(label_path);

        if (labels_.size() != image_count_) {
            throw std::runtime_error("MNIST image and label counts do not match");
        }
    }

    std::size_t size() const {
        return image_count_;
    }

    std::size_t batch_size() const {
        return batch_size_;
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

    void reset() {
        cursor_ = 0;
    }

    bool has_next() const {
        return cursor_ < image_count_;
    }

    MnistBatch next() {
        if (!has_next()) {
            throw std::runtime_error("MNIST dataloader has no batch left");
        }

        const std::size_t current_batch =
            std::min(batch_size_, image_count_ - cursor_);
        const std::size_t pixels = image_size();

        std::vector<float> batch_images(current_batch * pixels, 0.0f);
        std::vector<int64_t> batch_labels(current_batch, 0);

        for (std::size_t row = 0; row < current_batch; ++row) {
            const std::size_t source_image = cursor_ + row;
            const std::size_t source_offset = source_image * pixels;
            const std::size_t batch_offset = row * pixels;

            for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
                batch_images[batch_offset + pixel] =
                    static_cast<float>(images_[source_offset + pixel]) / 255.0f;
            }
            batch_labels[row] = labels_[source_image];
        }

        cursor_ += current_batch;

        return {
            dl::Tensor::from_host<float>(
                batch_images,
                dl::Shape({static_cast<int64_t>(current_batch),
                           static_cast<int64_t>(pixels)}),
                device_),
            dl::Tensor::from_host<int64_t>(
                batch_labels,
                dl::Shape({static_cast<int64_t>(current_batch)}),
                device_),
            current_batch,
        };
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

    std::size_t batch_size_ = 0;
    dl::Device device_;
    std::size_t cursor_ = 0;
    std::size_t image_count_ = 0;
    std::size_t rows_ = 0;
    std::size_t cols_ = 0;
    std::vector<uint8_t> images_;
    std::vector<int64_t> labels_;
};

}  // namespace demo::mnist
