#pragma once

#include <dl/core/Device.hpp>
#include <dl/data/Dataset.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

namespace dl::data {

template <typename Sample, typename Batch>
class DataLoader {
public:
    using CollateFn =
        std::function<Batch(const std::vector<Sample>&, const Device&)>;

    DataLoader(const Dataset<Sample>& dataset,
               std::size_t batch_size,
               Device device,
               CollateFn collate,
               bool shuffle = false,
               bool drop_last = false,
               uint64_t seed = 1234)
        : dataset_(dataset),
          batch_size_(batch_size),
          device_(device),
          collate_(std::move(collate)),
          shuffle_(shuffle),
          drop_last_(drop_last),
          seed_(seed) {
        if (batch_size_ == 0) {
            throw std::runtime_error("DataLoader batch size must be positive");
        }
        if (!collate_) {
            throw std::runtime_error("DataLoader collate function is empty");
        }
        reset();
    }

    std::size_t size() const {
        return dataset_.size();
    }

    std::size_t batch_size() const {
        return batch_size_;
    }

    const Device& device() const {
        return device_;
    }

    void reset() {
        order_.resize(dataset_.size());
        std::iota(order_.begin(), order_.end(), std::size_t{0});

        if (shuffle_) {
            std::mt19937_64 rng(seed_ + epoch_);
            std::shuffle(order_.begin(), order_.end(), rng);
            ++epoch_;
        }

        cursor_ = 0;
    }

    bool has_next() const {
        if (cursor_ >= order_.size()) {
            return false;
        }

        if (drop_last_ && order_.size() - cursor_ < batch_size_) {
            return false;
        }

        return true;
    }

    Batch next() {
        if (!has_next()) {
            throw std::runtime_error("DataLoader has no batch left");
        }

        const std::size_t end =
            std::min(cursor_ + batch_size_, order_.size());
        std::vector<Sample> samples;
        samples.reserve(end - cursor_);

        for (std::size_t i = cursor_; i < end; ++i) {
            samples.push_back(dataset_.get(order_[i]));
        }

        cursor_ = end;
        return collate_(samples, device_);
    }

private:
    const Dataset<Sample>& dataset_;
    std::size_t batch_size_ = 0;
    Device device_;
    CollateFn collate_;
    bool shuffle_ = false;
    bool drop_last_ = false;
    uint64_t seed_ = 1234;
    uint64_t epoch_ = 0;
    std::size_t cursor_ = 0;
    std::vector<std::size_t> order_;
};

}  // namespace dl::data
