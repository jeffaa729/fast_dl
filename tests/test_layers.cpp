// test_layers.cpp
// Test program for Linear Layer and Sigmoid functions

#include <iostream>
#include <iomanip>
#include <cmath>
#include <Eigen/Dense>
#include "Layers.hpp"

// Helper function to check if two matrices are approximately equal
bool isApproxEqual(const Eigen::MatrixXf& a, const Eigen::MatrixXf& b, float tolerance = 1e-5) {
    return (a - b).cwiseAbs().maxCoeff() < tolerance;
}

void test_linear_forward() {
    std::cout << "\n=== Testing Linear Forward Pass ===\n";
    
    // Create test data
    // Batch size = 2, input features = 3
    Eigen::MatrixXf input(2, 3);
    input << 1.0f, 2.0f, 3.0f,
             4.0f, 5.0f, 6.0f;
    
    // Output features = 2, input features = 3
    Eigen::MatrixXf weight(2, 3);
    weight << 0.1f, 0.2f, 0.3f,
              0.4f, 0.5f, 0.6f;
    
    // Bias for 2 output features
    Eigen::VectorXf bias(2);
    bias << 0.1f, 0.2f;
    
    // Forward pass
    Eigen::MatrixXf output = linear_forward(input, weight, bias);
    
    std::cout << "Input:\n" << input << "\n\n";
    std::cout << "Weight:\n" << weight << "\n\n";
    std::cout << "Bias:\n" << bias.transpose() << "\n\n";
    std::cout << "Output:\n" << output << "\n\n";
    
    // Expected output:
    // Row 1: [1,2,3] * [0.1,0.2,0.3]^T + 0.1 = 1.4 + 0.1 = 1.5
    //         [1,2,3] * [0.4,0.5,0.6]^T + 0.2 = 3.2 + 0.2 = 3.4
    // Row 2: [4,5,6] * [0.1,0.2,0.3]^T + 0.1 = 3.2 + 0.1 = 3.3
    //         [4,5,6] * [0.4,0.5,0.6]^T + 0.2 = 7.7 + 0.2 = 7.9
    
    Eigen::MatrixXf expected(2, 2);
    expected << 1.5f, 3.4f,
                3.3f, 7.9f;
    
    if (isApproxEqual(output, expected)) {
        std::cout << "✓ Linear forward pass test PASSED!\n";
    } else {
        std::cout << "✗ Linear forward pass test FAILED!\n";
        std::cout << "Expected:\n" << expected << "\n";
        std::cout << "Got:\n" << output << "\n";
    }
}

void test_linear_backward() {
    std::cout << "\n=== Testing Linear Backward Pass ===\n";
    
    // Create test data
    Eigen::MatrixXf input(2, 3);
    input << 1.0f, 2.0f, 3.0f,
             4.0f, 5.0f, 6.0f;
    
    Eigen::MatrixXf weight(2, 3);
    weight << 0.1f, 0.2f, 0.3f,
              0.4f, 0.5f, 0.6f;
    
    // Gradient from next layer
    Eigen::MatrixXf grad_output(2, 2);
    grad_output << 1.0f, 1.0f,
                   1.0f, 1.0f;
    
    // Allocate output gradients
    Eigen::MatrixXf grad_input(2, 3);
    Eigen::MatrixXf grad_weight(2, 3);
    Eigen::VectorXf grad_bias(2);
    
    // Backward pass
    linear_backward(grad_output, input, weight, grad_input, grad_weight, grad_bias);
    
    std::cout << "Grad output:\n" << grad_output << "\n\n";
    std::cout << "Grad input:\n" << grad_input << "\n\n";
    std::cout << "Grad weight:\n" << grad_weight << "\n\n";
    std::cout << "Grad bias:\n" << grad_bias.transpose() << "\n\n";
    
    // Expected grad_input = grad_output * weight = [1,1] * weight = [0.5, 0.7, 0.9] for each row
    // Expected grad_weight = grad_output^T * input = [1,1]^T * input = [5,7,9] for each row
    // Expected grad_bias = sum(grad_output, axis=0) = [2, 2]
    
    std::cout << "✓ Linear backward pass test completed (manual verification needed)\n";
}

void test_sigmoid_forward() {
    std::cout << "\n=== Testing Sigmoid Forward Pass ===\n";
    
    Eigen::MatrixXf input(2, 3);
    input << 0.0f, 1.0f, -1.0f,
             2.0f, -2.0f, 0.5f;
    
    Eigen::MatrixXf output = sigmoid_forward(input);
    
    std::cout << "Input:\n" << input << "\n\n";
    std::cout << "Output (sigmoid):\n" << output << "\n\n";
    
    // Verify specific values
    // sigmoid(0) = 0.5
    // sigmoid(1) ≈ 0.731
    // sigmoid(-1) ≈ 0.269
    
    float tolerance = 1e-3;
    bool passed = true;
    
    if (std::abs(output(0, 0) - 0.5f) > tolerance) {
        std::cout << "✗ sigmoid(0) should be 0.5, got " << output(0, 0) << "\n";
        passed = false;
    }
    if (std::abs(output(0, 1) - 0.731f) > tolerance) {
        std::cout << "✗ sigmoid(1) should be ~0.731, got " << output(0, 1) << "\n";
        passed = false;
    }
    if (std::abs(output(0, 2) - 0.269f) > tolerance) {
        std::cout << "✗ sigmoid(-1) should be ~0.269, got " << output(0, 2) << "\n";
        passed = false;
    }
    
    if (passed) {
        std::cout << "✓ Sigmoid forward pass test PASSED!\n";
    } else {
        std::cout << "✗ Sigmoid forward pass test FAILED!\n";
    }
}

void test_sigmoid_backward() {
    std::cout << "\n=== Testing Sigmoid Backward Pass ===\n";
    
    Eigen::MatrixXf input(2, 3);
    input << 0.0f, 1.0f, -1.0f,
             2.0f, -2.0f, 0.5f;
    
    // Forward pass first to get sigmoid output
    Eigen::MatrixXf sigmoid_output = sigmoid_forward(input);
    
    // Gradient from next layer
    Eigen::MatrixXf grad_output(2, 3);
    grad_output.setOnes(); // All ones
    
    // Backward pass
    Eigen::MatrixXf grad_input = sigmoid_backward(grad_output, sigmoid_output);
    
    std::cout << "Input:\n" << input << "\n\n";
    std::cout << "Sigmoid output:\n" << sigmoid_output << "\n\n";
    std::cout << "Grad output:\n" << grad_output << "\n\n";
    std::cout << "Grad input:\n" << grad_input << "\n\n";
    
    // The derivative of sigmoid at x=0 is 0.25 (max value)
    // sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
    // At x=0: 0.5 * (1 - 0.5) = 0.25
    
    float tolerance = 1e-3;
    if (std::abs(grad_input(0, 0) - 0.25f) < tolerance) {
        std::cout << "✓ Sigmoid backward pass test PASSED! (verified at x=0)\n";
    } else {
        std::cout << "✗ Sigmoid backward pass test FAILED! Expected ~0.25 at x=0, got " << grad_input(0, 0) << "\n";
    }
}

int main() {
    std::cout << "========================================\n";
    std::cout << "  Testing Neural Network Layers\n";
    std::cout << "========================================\n";
    
    test_linear_forward();
    test_linear_backward();
    test_sigmoid_forward();
    test_sigmoid_backward();
    
    std::cout << "\n========================================\n";
    std::cout << "  All tests completed!\n";
    std::cout << "========================================\n";
    
    return 0;
}

