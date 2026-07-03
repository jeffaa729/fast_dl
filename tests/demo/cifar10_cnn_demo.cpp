#include <dl/dl.hpp>

#include "cifar10/Cifar10DataLoader.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

class Cifar10CNN : public dl::nn::Module {
public:
    Cifar10CNN(const dl::Device& device,
               dl::nn::Conv2DInit conv_init,
               dl::nn::LinearInit linear_init)
        : conv1_(3, 16, 3, device, 1, 1, conv_init),
          relu_(),
          pool_(2),
          fc_(16 * 16 * 16, 10, device, linear_init) {
        register_module(conv1_);
        register_module(relu_);
        register_module(pool_);
        register_module(fc_);
    }

    dl::Tensor forward(const dl::Tensor& input) override {
        dl::Tensor x = conv1_(input);
        x = relu_(x);
        x = pool_(x);
        x = dl::ops::flatten(x);
        return fc_(x);
    }

private:
    dl::nn::Conv2D conv1_;
    dl::nn::ReLU relu_;
    dl::nn::MaxPool2D pool_;
    dl::nn::Linear fc_;
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

int env_non_negative_int(const char* name, int default_value) {
    const char* value = std::getenv(name);
    if (value == nullptr) {
        return default_value;
    }

    const int parsed = std::atoi(value);
    return parsed >= 0 ? parsed : default_value;
}

float env_float(const char* name, float default_value) {
    const char* value = std::getenv(name);
    if (value == nullptr) {
        return default_value;
    }

    const float parsed = std::strtof(value, nullptr);
    return parsed > 0.0f ? parsed : default_value;
}

dl::nn::LinearInit env_linear_init() {
    const char* value = std::getenv("CIFAR10_LINEAR_INIT");
    if (value == nullptr) {
        return dl::nn::LinearInit::KaimingUniform;
    }

    const std::string name(value);
    if (name == "xavier") {
        return dl::nn::LinearInit::XavierUniform;
    }
    return dl::nn::LinearInit::KaimingUniform;
}

dl::nn::Conv2DInit env_conv_init() {
    const char* value = std::getenv("CIFAR10_CONV_INIT");
    if (value == nullptr) {
        return dl::nn::Conv2DInit::KaimingUniform;
    }

    const std::string name(value);
    if (name == "xavier") {
        return dl::nn::Conv2DInit::XavierUniform;
    }
    return dl::nn::Conv2DInit::KaimingUniform;
}

const char* linear_init_name(dl::nn::LinearInit init) {
    switch (init) {
        case dl::nn::LinearInit::XavierUniform:
            return "xavier";
        case dl::nn::LinearInit::KaimingUniform:
            return "kaiming";
    }
    return "kaiming";
}

const char* conv_init_name(dl::nn::Conv2DInit init) {
    switch (init) {
        case dl::nn::Conv2DInit::XavierUniform:
            return "xavier";
        case dl::nn::Conv2DInit::KaimingUniform:
            return "kaiming";
    }
    return "kaiming";
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

Metrics evaluate(Cifar10CNN& model,
                 demo::cifar10::Cifar10DataLoader& loader,
                 int max_batches = 0) {
    dl::autograd::NoGradGuard no_grad;
    model.eval();
    loader.reset();

    double loss_sum = 0.0;
    std::size_t correct = 0;
    std::size_t total = 0;
    int batches = 0;

    while (loader.has_next() && (max_batches <= 0 || batches < max_batches)) {
        demo::cifar10::Cifar10Batch batch = loader.next();
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

int train_one_epoch(Cifar10CNN& model,
                    demo::cifar10::Cifar10DataLoader& loader,
                    dl::optim::SGD& optimizer,
                    int max_batches = 0) {
    model.train();
    loader.reset();

    int batches = 0;

    while (loader.has_next() && (max_batches <= 0 || batches < max_batches)) {
        demo::cifar10::Cifar10Batch batch = loader.next();

        optimizer.zero_grad();
        dl::Tensor logits = model(batch.images);
        dl::Tensor loss = dl::ops::cross_entropy(logits, batch.labels);

        loss.backward();
        optimizer.step();
        ++batches;
    }

    return batches;
}

void print_sample_results(Cifar10CNN& model,
                          demo::cifar10::Cifar10DataLoader& loader,
                          int sample_count) {
    dl::autograd::NoGradGuard no_grad;
    model.eval();
    loader.reset();

    demo::cifar10::Cifar10Batch batch = loader.next();
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
        const int epochs = env_int("CIFAR10_EPOCHS", 1);
        const int batch_size = env_int("CIFAR10_BATCH_SIZE", 64);
        const int max_train_batches = env_non_negative_int("CIFAR10_MAX_TRAIN_BATCHES", 10);
        const int max_test_batches = env_non_negative_int("CIFAR10_MAX_TEST_BATCHES", 5);
        const int train_eval_batches = env_non_negative_int("CIFAR10_TRAIN_EVAL_BATCHES", 5);
        const int sample_count = env_int("CIFAR10_SAMPLES", 3);
        const float learning_rate = env_float("CIFAR10_LR", 0.01f);
        const dl::nn::Conv2DInit conv_init = env_conv_init();
        const dl::nn::LinearInit linear_init = env_linear_init();

        const dl::Device device(dl::DeviceType::CUDA, 0);
        demo::cifar10::Cifar10DataLoader train_loader(
            "data/cifar-10-batches-bin",
            true,
            static_cast<std::size_t>(batch_size),
            device);
        demo::cifar10::Cifar10DataLoader test_loader(
            "data/cifar-10-batches-bin",
            false,
            static_cast<std::size_t>(batch_size),
            device);

        Cifar10CNN model(device, conv_init, linear_init);
        dl::optim::SGD optimizer(model.parameters(), learning_rate);

        std::cout << "cifar10_cnn_demo : start\n";
        std::cout << "config : epochs=" << epochs
                  << " batch_size=" << batch_size
                  << " lr=" << learning_rate
                  << " conv_init=" << conv_init_name(conv_init)
                  << " linear_init=" << linear_init_name(linear_init)
                  << " max_train_batches=" << max_train_batches
                  << " max_test_batches=" << max_test_batches
                  << " train_eval_batches=" << train_eval_batches
                  << " train_samples=" << train_loader.size()
                  << " test_samples=" << test_loader.size() << "\n";

        Metrics last_train;
        Metrics last_test;
        for (int epoch = 1; epoch <= epochs; ++epoch) {
            const int trained_batches = train_one_epoch(
                model, train_loader, optimizer, max_train_batches);
            last_train = evaluate(model, train_loader, train_eval_batches);
            last_test = evaluate(model, test_loader, max_test_batches);

            std::cout << std::fixed << std::setprecision(4)
                      << "epoch " << epoch
                      << " : batches=" << trained_batches
                      << " train_loss=" << last_train.loss
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
        std::cout << "cifar10_cnn_demo_error : " << error.what() << "\n";
        passed = false;
    } catch (...) {
        passed = false;
    }

    std::cout << "cifar10_cnn_demo : "
              << (passed ? "passed" : "not passed") << "\n";

    return passed ? 0 : 1;
}
