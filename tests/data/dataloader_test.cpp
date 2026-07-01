#include <dl/dl.hpp>

#include <cstddef>
#include <iostream>
#include <utility>
#include <vector>

namespace {

class IntDataset : public dl::data::Dataset<int> {
public:
    explicit IntDataset(std::vector<int> values) : values_(std::move(values)) {}

    std::size_t size() const override {
        return values_.size();
    }

    int get(std::size_t index) const override {
        return values_.at(index);
    }

private:
    std::vector<int> values_;
};

struct IntBatch {
    std::vector<int> values;
    dl::Device device;
};

IntBatch collate_ints(const std::vector<int>& samples,
                      const dl::Device& device) {
    return {samples, device};
}

bool same_values(const std::vector<int>& actual,
                 const std::vector<int>& expected) {
    return actual == expected;
}

}  // namespace

int main() {
    bool passed = true;

    try {
        const dl::Device device(dl::DeviceType::CUDA, 0);
        const IntDataset dataset({0, 1, 2, 3, 4});

        dl::data::DataLoader<int, IntBatch> loader(
            dataset,
            2,
            device,
            collate_ints);

        IntBatch first = loader.next();
        IntBatch second = loader.next();
        IntBatch third = loader.next();

        passed = passed &&
                 loader.size() == 5 &&
                 loader.batch_size() == 2 &&
                 first.device.is_cuda() &&
                 same_values(first.values, {0, 1}) &&
                 same_values(second.values, {2, 3}) &&
                 same_values(third.values, {4}) &&
                 !loader.has_next();

        loader.reset();
        IntBatch reset_first = loader.next();
        passed = passed && same_values(reset_first.values, {0, 1});

        dl::data::DataLoader<int, IntBatch> drop_last_loader(
            dataset,
            2,
            device,
            collate_ints,
            false,
            true);

        IntBatch drop_first = drop_last_loader.next();
        IntBatch drop_second = drop_last_loader.next();
        passed = passed &&
                 same_values(drop_first.values, {0, 1}) &&
                 same_values(drop_second.values, {2, 3}) &&
                 !drop_last_loader.has_next();
    } catch (...) {
        passed = false;
    }

    std::cout << "dataloader_test : "
              << (passed ? "passed" : "not passed") << "\n";

    return passed ? 0 : 1;
}
