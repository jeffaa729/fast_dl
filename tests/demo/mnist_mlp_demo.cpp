#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <vector>

#include <dl/dl.hpp>

#include "mnist/MnistDataLoader.hpp"

namespace {

class MnistMLP : public dl::nn::Module {
public:
    MnistMLP(const dl::Device& device, dl::nn::LinearInit init)
        : fc1_(784, 128, device, init),
          relu_(),
          fc2_(128, 10, device, init) {
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

const char* linear_init_name(dl::nn::LinearInit init) {
    switch (init) {
        case dl::nn::LinearInit::XavierUniform:
            return "xavier";
        case dl::nn::LinearInit::KaimingUniform:
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

int train_one_epoch(MnistMLP& model,
                    demo::mnist::MnistDataLoader& loader,
                    dl::optim::SGD& optimizer,
                    int max_batches = 0) {
    model.train();
    loader.reset();

    int batches = 0;

    while (loader.has_next() && (max_batches <= 0 || batches < max_batches)) {
        demo::mnist::MnistBatch batch = loader.next();

        optimizer.zero_grad();
        dl::Tensor logits = model(batch.images);
        dl::Tensor loss = dl::ops::cross_entropy(logits, batch.labels);

        loss.backward();
        optimizer.step();
        ++batches;
    }

    return batches;
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
        constexpr int epochs = 20;
        constexpr int batch_size = 64;
        constexpr int max_train_batches = 0;
        constexpr int max_test_batches = 0;
        constexpr int train_eval_batches = 0;
        constexpr int sample_count = 3;
        constexpr float learning_rate = 0.01f;
        constexpr dl::nn::LinearInit linear_init = dl::nn::LinearInit::KaimingUniform;

        const dl::Device device(dl::DeviceType::CUDA, 0);
        demo::mnist::MnistDataLoader train_loader(
            "data/mnist/train-images-idx3-ubyte",
            "data/mnist/train-labels-idx1-ubyte",
            static_cast<std::size_t>(batch_size),
            device);
        demo::mnist::MnistDataLoader test_loader(
            "data/mnist/t10k-images-idx3-ubyte",
            "data/mnist/t10k-labels-idx1-ubyte",
            static_cast<std::size_t>(batch_size),
            device);

        MnistMLP model(device, linear_init);
        dl::optim::SGD optimizer(model.parameters(), learning_rate);

        std::cout << "mnist_mlp_demo : start\n";
        model.print_parameters();
        std::cout << "config : epochs=" << epochs
                  << " batch_size=" << batch_size
                  << " lr=" << learning_rate
                  << " init=" << linear_init_name(linear_init)
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
        std::cout << "mnist_mlp_demo_error : " << error.what() << "\n";
        passed = false;
    } catch (...) {
        passed = false;
    }

    std::cout << "mnist_mlp_demo : "
              << (passed ? "passed" : "not passed") << "\n";

    return passed ? 0 : 1;
}
