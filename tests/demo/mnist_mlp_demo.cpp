#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <dl/dl.hpp>

#include "mnist/MnistDataLoader.hpp"

namespace {

class MnistMLP : public dl::nn::Module {
public:
    explicit MnistMLP(const dl::Device& device)
        : fc1_(784, 128, device),
          relu_(),
          fc2_(128, 10, device) {
        register_module(fc1_);
        register_module(relu_);
        register_module(fc2_);
    }

    dl::Tensor forward(const dl::Tensor& input) override {
        return fc2_(relu_(fc1_(input)));
    }

private:
    dl::nn::Linear fc1_;
    dl::nn::ReLU relu_;
    dl::nn::Linear fc2_;
};

struct Metrics {
    float loss = 0.0f;
    float accuracy = 0.0f;
};

int env_int(const char* name, int default_value) {
    const char* value = std::getenv(name);
    if (value == nullptr) {
        return default_value;
    }

    const int parsed = std::atoi(value);
    return parsed > 0 ? parsed : default_value;
}

float env_float(const char* name, float default_value) {
    const char* value = std::getenv(name);
    if (value == nullptr) {
        return default_value;
    }

    const float parsed = std::strtof(value, nullptr);
    return parsed > 0.0f ? parsed : default_value;
}

std::vector<int64_t> predict_classes(const dl::Tensor& logits) {
    const std::vector<float> logits_host = logits.to_host<float>();
    const std::size_t batch = static_cast<std::size_t>(logits.shape()[0]);
    const std::size_t classes = static_cast<std::size_t>(logits.shape()[1]);
    std::vector<int64_t> predictions(batch, 0);

    for (std::size_t row = 0; row < batch; ++row) {
        const std::size_t offset = row * classes;
        std::size_t best_class = 0;
        float best_logit = logits_host[offset];

        for (std::size_t col = 1; col < classes; ++col) {
            const float value = logits_host[offset + col];
            if (value > best_logit) {
                best_logit = value;
                best_class = col;
            }
        }

        predictions[row] = static_cast<int64_t>(best_class);
    }

    return predictions;
}

std::size_t count_correct(const dl::Tensor& logits, const dl::Tensor& labels) {
    const std::vector<int64_t> predictions = predict_classes(logits);
    const std::vector<int64_t> labels_host = labels.to_host<int64_t>();
    std::size_t correct = 0;

    for (std::size_t i = 0; i < labels_host.size(); ++i) {
        if (predictions[i] == labels_host[i]) {
            ++correct;
        }
    }

    return correct;
}

Metrics evaluate(MnistMLP& model,
                 demo::mnist::MnistDataLoader& loader,
                 int max_batches = 0) {
    dl::autograd::NoGradGuard no_grad;
    model.eval();
    loader.reset();

    double loss_sum = 0.0;
    std::size_t correct = 0;
    std::size_t total = 0;
    int batches = 0;

    while (loader.has_next() && (max_batches <= 0 || batches < max_batches)) {
        demo::mnist::MnistBatch batch = loader.next();
        dl::Tensor logits = model(batch.images);
        dl::Tensor loss = dl::ops::cross_entropy(logits, batch.labels);

        loss_sum += static_cast<double>(loss.to_host<float>()[0]) *
                    static_cast<double>(batch.size);
        correct += count_correct(logits, batch.labels);
        total += batch.size;
        ++batches;
    }

    model.train();

    Metrics metrics;
    if (total > 0) {
        metrics.loss = static_cast<float>(loss_sum / static_cast<double>(total));
        metrics.accuracy =
            static_cast<float>(correct) / static_cast<float>(total);
    }
    return metrics;
}

Metrics train_one_epoch(MnistMLP& model,
                        demo::mnist::MnistDataLoader& loader,
                        dl::optim::SGD& optimizer,
                        int max_batches = 0) {
    model.train();
    loader.reset();

    double loss_sum = 0.0;
    std::size_t correct = 0;
    std::size_t total = 0;
    int batches = 0;

    while (loader.has_next() && (max_batches <= 0 || batches < max_batches)) {
        demo::mnist::MnistBatch batch = loader.next();

        optimizer.zero_grad();
        dl::Tensor logits = model(batch.images);
        dl::Tensor loss = dl::ops::cross_entropy(logits, batch.labels);
        const float loss_value = loss.to_host<float>()[0];

        loss.backward();
        optimizer.step();

        loss_sum += static_cast<double>(loss_value) *
                    static_cast<double>(batch.size);
        correct += count_correct(logits, batch.labels);
        total += batch.size;
        ++batches;
    }

    Metrics metrics;
    if (total > 0) {
        metrics.loss = static_cast<float>(loss_sum / static_cast<double>(total));
        metrics.accuracy =
            static_cast<float>(correct) / static_cast<float>(total);
    }
    return metrics;
}

void print_sample_results(MnistMLP& model,
                          demo::mnist::MnistDataLoader& loader,
                          int sample_count) {
    dl::autograd::NoGradGuard no_grad;
    model.eval();
    loader.reset();

    demo::mnist::MnistBatch batch = loader.next();
    const std::vector<int64_t> predictions = predict_classes(model(batch.images));
    const std::vector<int64_t> labels = batch.labels.to_host<int64_t>();

    const int count = std::min(
        sample_count,
        static_cast<int>(std::min(predictions.size(), labels.size())));

    std::cout << "samples :";
    for (int i = 0; i < count; ++i) {
        std::cout << " [" << i << "] pred=" << predictions[static_cast<std::size_t>(i)]
                  << " label=" << labels[static_cast<std::size_t>(i)];
    }
    std::cout << "\n";

    model.train();
}

}  // namespace

int main() {
    bool passed = true;

    try {
        const int epochs = env_int("MNIST_EPOCHS", 100);
        const int batch_size = env_int("MNIST_BATCH_SIZE", 64);
        const int max_train_batches = env_int("MNIST_MAX_TRAIN_BATCHES", 0);
        const int max_test_batches = env_int("MNIST_MAX_TEST_BATCHES", 0);
        const int sample_count = env_int("MNIST_SAMPLES", 3);
        const float learning_rate = env_float("MNIST_LR", 0.01f);

        const dl::Device device(dl::DeviceType::CUDA, 0);
        demo::mnist::MnistDataLoader train_loader(
            "mnist/train-images-idx3-ubyte",
            "mnist/train-labels-idx1-ubyte",
            static_cast<std::size_t>(batch_size),
            device);
        demo::mnist::MnistDataLoader test_loader(
            "mnist/t10k-images-idx3-ubyte",
            "mnist/t10k-labels-idx1-ubyte",
            static_cast<std::size_t>(batch_size),
            device);

        MnistMLP model(device);
        dl::optim::SGD optimizer(model.parameters(), learning_rate);

        std::cout << "mnist_mlp_demo : start\n";
        std::cout << "config : epochs=" << epochs
                  << " batch_size=" << batch_size
                  << " lr=" << learning_rate
                  << " train_samples=" << train_loader.size()
                  << " test_samples=" << test_loader.size() << "\n";

        Metrics last_train;
        Metrics last_test;
        for (int epoch = 1; epoch <= epochs; ++epoch) {
            last_train = train_one_epoch(model, train_loader, optimizer,
                                         max_train_batches);
            last_test = evaluate(model, test_loader, max_test_batches);

            std::cout << std::fixed << std::setprecision(4)
                      << "epoch " << epoch
                      << " : train_loss=" << last_train.loss
                      << " train_acc=" << last_train.accuracy
                      << " test_loss=" << last_test.loss
                      << " test_acc=" << last_test.accuracy << "\n";
        }

        print_sample_results(model, test_loader, sample_count);

        passed = std::isfinite(last_train.loss) &&
                 std::isfinite(last_test.loss) &&
                 last_train.accuracy >= 0.0f &&
                 last_train.accuracy <= 1.0f &&
                 last_test.accuracy >= 0.0f &&
                 last_test.accuracy <= 1.0f;
    } catch (const std::exception& error) {
        std::cout << "mnist_mlp_demo_error : " << error.what() << "\n";
        passed = false;
    } catch (...) {
        passed = false;
    }

    std::cout << "mnist_mlp_demo : "
              << (passed ? "passed" : "not passed") << "\n";

    return passed ? 0 : 1;
}
