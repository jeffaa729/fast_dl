// test_oop.cpp
// Tests for OOP layers and SGD optimizer

#include <iostream>
#include <Eigen/Dense>

#include "LinearLayer.hpp"
#include "SigmoidLayer.hpp"
#include "Optimizer.hpp"

// Helper to check approximate equality
bool isApproxEqual(const Eigen::MatrixXf& a, const Eigen::MatrixXf& b, float tol = 1e-5f) {
    return (a - b).cwiseAbs().maxCoeff() < tol;
}

void test_linear_forward_backward() {
    std::cout << "\n=== OOP LinearLayer: forward/backward ===\n";

    // input: (2,3)
    Eigen::MatrixXf input(2, 3);
    input << 1, 2, 3,
             4, 5, 6;

    // create layer and set deterministic weights/bias for testing
    LinearLayer layer(3, 2);
    layer.weights() << 0.1f, 0.2f, 0.3f,
                       0.4f, 0.5f, 0.6f;
    layer.biases() << 0.1f, 0.2f;

    // forward
    Eigen::MatrixXf out = layer.forward(input);
    Eigen::MatrixXf expected_out(2, 2);
    expected_out << 1.5f, 3.4f,
                    3.3f, 7.9f;

    std::cout << "Output:\n" << out << "\n";
    if (isApproxEqual(out, expected_out)) {
        std::cout << "✓ Linear forward OK\n";
    } else {
        std::cout << "✗ Linear forward mismatch\n";
    }

    // backward with grad_output of ones
    Eigen::MatrixXf grad_out = Eigen::MatrixXf::Ones(2, 2);
    Eigen::MatrixXf grad_in = layer.backward(grad_out);

    Eigen::MatrixXf expected_grad_in(2, 3);
    expected_grad_in << 0.5f, 0.7f, 0.9f,
                        0.5f, 0.7f, 0.9f;

    std::cout << "Grad input:\n" << grad_in << "\n";
    if (isApproxEqual(grad_in, expected_grad_in)) {
        std::cout << "✓ Linear backward grad_input OK\n";
    } else {
        std::cout << "✗ Linear backward grad_input mismatch\n";
    }

    Eigen::MatrixXf expected_grad_w(2, 3);
    expected_grad_w << 5.f, 7.f, 9.f,
                       5.f, 7.f, 9.f;
    Eigen::VectorXf expected_grad_b(2);
    expected_grad_b << 2.f, 2.f;

    std::cout << "Grad weight:\n" << layer.grad_weights() << "\n";
    std::cout << "Grad bias:\n" << layer.grad_biases().transpose() << "\n";
    if (isApproxEqual(layer.grad_weights(), expected_grad_w) &&
        (layer.grad_biases() - expected_grad_b).cwiseAbs().maxCoeff() < 1e-5f) {
        std::cout << "✓ Linear backward params OK\n";
    } else {
        std::cout << "✗ Linear backward params mismatch\n";
    }
}

void test_sigmoid_forward_backward() {
    std::cout << "\n=== OOP SigmoidLayer: forward/backward ===\n";

    Eigen::MatrixXf input(2, 3);
    input << 0.0f, 1.0f, -1.0f,
             2.0f, -2.0f, 0.5f;

    SigmoidLayer layer;
    Eigen::MatrixXf out = layer.forward(input);

    std::cout << "Output:\n" << out << "\n";
    float tol = 1e-3f;
    bool ok = true;
    if (std::abs(out(0, 0) - 0.5f) > tol) ok = false;
    if (std::abs(out(0, 1) - 0.731f) > tol) ok = false;
    if (std::abs(out(0, 2) - 0.269f) > tol) ok = false;
    std::cout << (ok ? "✓ Sigmoid forward OK\n" : "✗ Sigmoid forward mismatch\n");

    Eigen::MatrixXf grad_out = Eigen::MatrixXf::Ones(2, 3);
    Eigen::MatrixXf grad_in = layer.backward(grad_out);
    std::cout << "Grad input:\n" << grad_in << "\n";
    if (std::abs(grad_in(0, 0) - 0.25f) < tol) {
        std::cout << "✓ Sigmoid backward OK at x=0\n";
    } else {
        std::cout << "✗ Sigmoid backward mismatch at x=0\n";
    }
}

void test_sgd_update() {
    std::cout << "\n=== SGD Optimizer update ===\n";

    LinearLayer layer(3, 2);
    layer.weights() << 0.1f, 0.2f, 0.3f,
                       0.4f, 0.5f, 0.6f;
    layer.biases() << 0.1f, 0.2f;

    Eigen::MatrixXf input(2, 3);
    input << 1, 2, 3,
             4, 5, 6;

    Eigen::MatrixXf grad_out = Eigen::MatrixXf::Ones(2, 2);
    layer.forward(input);
    layer.backward(grad_out);

    SGD optimizer(0.1f);
    optimizer.step(layer);

    Eigen::MatrixXf expected_w(2, 3);
    expected_w << 0.1f, 0.2f, 0.3f,
                  0.4f, 0.5f, 0.6f;
    Eigen::VectorXf expected_b(2);
    expected_b << 0.1f, 0.2f;

    // expected new params = old - lr * grad
    Eigen::MatrixXf expected_new_w = expected_w - 0.1f * layer.grad_weights();
    Eigen::VectorXf expected_new_b = expected_b - 0.1f * layer.grad_biases();

    std::cout << "Updated weight:\n" << layer.weights() << "\n";
    std::cout << "Updated bias:\n" << layer.biases().transpose() << "\n";

    bool ok = isApproxEqual(layer.weights(), expected_new_w) &&
              (layer.biases() - expected_new_b).cwiseAbs().maxCoeff() < 1e-5f;
    std::cout << (ok ? "✓ SGD update OK\n" : "✗ SGD update mismatch\n");
}

int main() {
    test_linear_forward_backward();
    test_sigmoid_forward_backward();
    test_sgd_update();
    return 0;
}
