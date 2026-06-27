#pragma once

#include <memory>
#include <vector>

#include <dl/tensor/Tensor.hpp>

namespace dl{

    class TensorImpl;
    namespace autograd {

        class Operator {
        public:
            virtual ~Operator() = default;
            void setup_computation_graph(const std::vector<Tensor>& inputs, const std::vector<Tensor>& outputs);
            virtual std::vector<Tensor> backward(const std::vector<Tensor>& grad_outputs) = 0;
            const std::vector<Tensor>& inputs() const;
            std::vector<Tensor> grad_outputs() const;
            int generation() const;

        
        private:
            std::vector<Tensor> inputs_;
            std::vector<std::weak_ptr<TensorImpl>> outputs_;
            int generation_ = 0;
        };

    }  // namespace autograd
}// namespace dl
