#include <dl/autograd/Operator.hpp>

#include <algorithm>
#include <stdexcept>

#include <dl/tensor/TensorImpl.hpp>

namespace dl::autograd {
    void Operator::setup_computation_graph(const std::vector<Tensor>& inputs, const std::vector<Tensor>& outputs) {
        inputs_ = inputs;
        generation_ = 0;
        for (const auto& input : inputs) {
            if (!input.impl()) {
                throw std::runtime_error("Input tensor must have a valid implementation");
            }
            generation_ = std::max(generation_, input.impl()->generation());
        }
        outputs_.clear();
        for (const auto& output : outputs) {
            if (!output.impl()) {
                throw std::runtime_error("Output tensor must have a valid implementation");
            }
            outputs_.push_back(output.impl());
        }
    }
    
    const std::vector<Tensor>& Operator::inputs() const {
        return inputs_;
    }

    int Operator::generation() const {
        return generation_;
    }

    std::vector<Tensor> Operator::grad_outputs() const {
        std::vector<Tensor> result;

        for (const auto& weak_output : outputs_) {
            auto output = weak_output.lock();
            if (!output) {
                continue;
            }

            Tensor grad = output->grad();
            if (grad.defined()) {
                result.push_back(grad);
            }
        }

        return result;
    }
} // namespace dl::autograd