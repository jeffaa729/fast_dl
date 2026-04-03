// test_network.cpp
// Integration test for Network + layers + SoftmaxCrossEntropyLoss + SGD

#include <iostream>
#include <Eigen/Dense>

#include "LinearLayer.hpp"
#include "SigmoidLayer.hpp"
#include "SoftmaxCrossEntropyLoss.hpp"
#include "Network.hpp"
#include "Optimizer.hpp"

void test_softmax_ce_standalone() {
    std::cout << "\n=== SoftmaxCrossEntropyLoss standalone test ===\n";

    // Two samples, three classes
    Eigen::MatrixXf logits(2, 3);
    logits << 1.0f, 2.0f, 3.0f,
              1.0f, 3.0f, 2.0f;
    Eigen::VectorXi labels(2);
    labels << 2, 1; // correct classes

    SoftmaxCrossEntropyLoss loss;
    float value = loss.forward(logits, labels);
    Eigen::MatrixXf grad = loss.backward();

    std::cout << "Loss value: " << value << "\n";
    std::cout << "Gradients:\n" << grad << "\n";

    // Basic sanity checks
    if (value > 0.0f && std::isfinite(value)) {
        std::cout << "✓ Loss is positive and finite\n";
    } else {
        std::cout << "✗ Loss value invalid\n";
    }
}

void test_network_training() {
    std::cout << "\n=== Network training test (loss should decrease) ===\n";

    // Tiny synthetic dataset: 4 samples, 4 features, 2 classes
    Eigen::MatrixXf X(4, 4);
    X << 1, 0, 0, 0,
         0, 1, 0, 0,
         0, 0, 1, 0,
         0, 0, 0, 1;
    Eigen::VectorXi y(4);
    y << 0, 0, 1, 1;

    // Build a simple network: 4 -> 4 -> 2
    Network net;
    SGD optimizer(0.1f);
    net.set_optimizer(&optimizer);
    net.add_layer(std::make_unique<LinearLayer>(4, 4));
    net.add_layer(std::make_unique<SigmoidLayer>());
    net.add_layer(std::make_unique<LinearLayer>(4, 2));

    const int epochs = 50;
    float first_loss = 0.0f;
    float last_loss = 0.0f;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        net.forward(X);
        float loss = net.compute_loss(y);
        net.backward();
        net.update();

        if (epoch == 0) first_loss = loss;
        last_loss = loss;

        if (epoch % 10 == 0 || epoch == epochs - 1) {
            std::cout << "Epoch " << epoch << " - loss: " << loss << "\n";
        }
    }

    if (last_loss < first_loss) {
        std::cout << "✓ Training loss decreased from " << first_loss
                  << " to " << last_loss << "\n";
    } else {
        std::cout << "✗ Training loss did not decrease\n";
    }
}

int main() {
    test_softmax_ce_standalone();
    test_network_training();
    return 0;
}

