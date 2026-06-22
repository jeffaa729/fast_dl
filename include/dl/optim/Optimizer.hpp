#pragma once

#include "Layer.hpp"
#include <unordered_map>

class Optimizer {
public:
    virtual ~Optimizer() = default;
    virtual void step(Layer& layer) = 0;
};

class SGD : public Optimizer {
public:
    explicit SGD(float lr) : learning_rate(lr) {}

    void step(Layer& layer) override {
        if (!layer.has_parameters()) return;

        auto params = layer.parameters();
        auto grads = layer.parameter_grads();
        auto bias_params = layer.bias_parameters();
        auto bias_grads = layer.bias_grads();

        for (std::size_t i = 0; i < params.size() && i < grads.size(); ++i) {
            if (params[i] && grads[i]) {
                *params[i] -= learning_rate * (*grads[i]);
            }
        }

        for (std::size_t i = 0; i < bias_params.size() && i < bias_grads.size(); ++i) {
            if (bias_params[i] && bias_grads[i]) {
                *bias_params[i] -= learning_rate * (*bias_grads[i]);
            }
        }
    }

private:
    float learning_rate;
};

class Adam : public Optimizer {
public:
    Adam(float lr = 1e-3f, float beta1 = 0.9f, float beta2 = 0.999f, float eps   = 1e-8f)
        : learning_rate(lr), beta1(beta1), beta2(beta2), epsilon(eps) {}

    void step(Layer& layer) override {
        if (!layer.has_parameters()) return;

        auto params      = layer.parameters();
        auto grads       = layer.parameter_grads();
        auto bias_params = layer.bias_parameters();
        auto bias_grads  = layer.bias_grads();

        for (std::size_t i = 0; i < params.size() && i < grads.size(); ++i) {
            auto* p = params[i];
            auto* g = grads[i];
            if (!p || !g) continue;

            AdamMatState& st = mat_states[p];

            if (st.m.size() == 0) {
                st.m = Eigen::MatrixXf::Zero(p->rows(), p->cols());
                st.v = Eigen::MatrixXf::Zero(p->rows(), p->cols());
            }

            st.m = beta1 * st.m + (1.0f - beta1) * (*g);
            st.v = beta2 * st.v + (1.0f - beta2) * g->array().square().matrix();

            Eigen::MatrixXf update =
                (st.m.array() / (st.v.array().sqrt() + epsilon)).matrix();

            *p -= learning_rate * update;
        }

        for (std::size_t i = 0; i < bias_params.size() && i < bias_grads.size(); ++i) {
            auto* p = bias_params[i];
            auto* g = bias_grads[i];
            if (!p || !g) continue;

            AdamVecState& st = vec_states[p];

            if (st.m.size() == 0) {
                st.m = Eigen::VectorXf::Zero(p->size());
                st.v = Eigen::VectorXf::Zero(p->size());
            }

            st.m = beta1 * st.m + (1.0f - beta1) * (*g);
            st.v = beta2 * st.v + (1.0f - beta2) * g->array().square().matrix();

            Eigen::VectorXf update =
                st.m.array() / (st.v.array().sqrt() + epsilon);

            *p -= learning_rate * update;
        }
    }

private:
    struct AdamMatState {
        Eigen::MatrixXf m;
        Eigen::MatrixXf v;
    };
    struct AdamVecState {
        Eigen::VectorXf m;
        Eigen::VectorXf v;
    };

    float learning_rate;
    float beta1;
    float beta2;
    float epsilon;

    std::unordered_map<Eigen::MatrixXf*, AdamMatState> mat_states;
    std::unordered_map<Eigen::VectorXf*, AdamVecState> vec_states;
};
