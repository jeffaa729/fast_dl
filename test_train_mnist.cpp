#include <iostream>
#include <Eigen/Dense>

#include "LinearLayer.hpp"
#include "SigmoidLayer.hpp"
#include "ParametricReLU.hpp"
#include "SoftmaxCrossEntropyLoss.hpp"
#include "Network.hpp"
#include "Optimizer.hpp"
#include "DataLoader.hpp"

void test_train_mnist(Optimizer &optimizer, const int epochs, const int batch_size) {
    DataLoader train(
        "mnist/train-images-idx3-ubyte",
        "mnist/train-labels-idx1-ubyte"
    );
    DataLoader test(
        "mnist/t10k-images-idx3-ubyte",
        "mnist/t10k-labels-idx1-ubyte", 
        300
    );
     
    int n    = train.getSize();

    const Eigen::MatrixXf& X_all = train.getImagesMatrix(); 
    int input_dim   = X_all.cols(); 
    int num_classes = 10;

    Network net;
    net.set_optimizer(&optimizer);

    net.add_layer(std::make_unique<LinearLayer>(input_dim, 128));
    net.add_layer(std::make_unique<ParametricReLU>(128, 0.1f));
    net.add_layer(std::make_unique<LinearLayer>(128, num_classes));



    for (int epoch = 0; epoch < epochs; ++epoch) {
        float epoch_loss_sum = 0.0f;
        int   batches_in_epoch = 0;

        for (int start = 0; start < n; start += batch_size) {
            int end    = std::min(start + batch_size, n);
            int b_size = end - start;  

            Eigen::MatrixXf X_batch = X_all.block(start, 0, b_size, input_dim);

            Eigen::VectorXi y_batch(b_size);
            for (int i = 0; i < b_size; ++i) {
                y_batch(i) = train.getLabels(start + i);
            }

            net.forward(X_batch);
            float loss = net.compute_loss(y_batch);
            net.backward();
            net.update();

            epoch_loss_sum += loss;
            ++batches_in_epoch;
        }

        float avg_loss = epoch_loss_sum / static_cast<float>(batches_in_epoch);

        if (epoch % 5 == 0) {
            std::cout << "Epoch " << epoch << " - avg loss: " << avg_loss << "\n";
        }
    }

    int test_n  = test.getSize();
    const Eigen::MatrixXf& X_test_all  = test.getImagesMatrix();
    Eigen::MatrixXf X_test = X_test_all.topRows(test_n);
    Eigen::VectorXi y_test(test_n);
    for (int i = 0; i < test_n; ++i) {
        y_test(i) = test.getLabels(i);
    }

    net.forward(X_test);
    float test_loss = net.compute_loss(y_test);

    // Get logits for accuracy
    const Eigen::MatrixXf& logits = net.get_cached_logits();  

    int correct = 0;
    for (int i = 0; i < test_n; ++i) {
        Eigen::MatrixXf::Index max_idx;
        logits.row(i).maxCoeff(&max_idx);
        int pred = static_cast<int>(max_idx);
        if (pred == y_test(i)) {
            ++correct;
        }
    }
    float test_acc = static_cast<float>(correct) / static_cast<float>(test_n);

    std::cout << "Final test loss: " << test_loss
              << " | Final test accuracy: " << test_acc << "\n";

}

int main() {
    const int batch_size = 128;
    const int epochs     = 30;
    SGD sgd_optimizer(0.1f);
    Adam adam_optimizer(1e-3f);
    std::cout << "using Adam optimizer: " << std::endl;
    test_train_mnist(adam_optimizer, epochs, batch_size);
    std::cout << "using SGD optimizer: " << std::endl;
    test_train_mnist(sgd_optimizer, epochs, batch_size);
    return 0;
}
