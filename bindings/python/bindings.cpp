#include <dl/dl.hpp>

#include <dl/core/CudaUtils.hpp>

#include <cuda_runtime.h>

#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>

#include <memory>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

dl::Shape make_shape(const std::vector<int64_t>& dims) {
    return dl::Shape(dims);
}

std::vector<int64_t> shape_dims(const dl::Shape& shape) {
    return shape.dims;
}

dl::Tensor tensor_from_float_list(const std::vector<float>& values,
                                  const std::vector<int64_t>& dims,
                                  const dl::Device& device) {
    return dl::Tensor::from_host<float>(values, make_shape(dims), device);
}

dl::Tensor tensor_from_int64_list(const std::vector<int64_t>& values,
                                  const std::vector<int64_t>& dims,
                                  const dl::Device& device) {
    return dl::Tensor::from_host<int64_t>(values, make_shape(dims), device);
}

std::vector<float> tensor_to_float_list(const dl::Tensor& tensor) {
    return tensor.to_host<float>();
}

std::vector<int64_t> tensor_to_int64_list(const dl::Tensor& tensor) {
    return tensor.to_host<int64_t>();
}

std::string tensor_repr(const dl::Tensor& tensor) {
    std::ostringstream os;
    os << "dl.Tensor(shape=" << tensor.shape().str()
       << ", dtype=" << dl::dtype_name(tensor.dtype())
       << ", device=" << tensor.device().str() << ")";
    return os.str();
}

std::vector<dl::Tensor> module_parameters(dl::nn::Module& module) {
    std::vector<dl::Tensor> out;
    for (dl::Tensor* parameter : module.parameters()) {
        if (parameter != nullptr) {
            out.push_back(*parameter);
        }
    }
    return out;
}

class PySGD {
public:
    PySGD(std::vector<dl::Tensor> parameters, float learning_rate)
        : parameters_(std::move(parameters)) {
        std::vector<dl::Tensor*> parameter_ptrs;
        parameter_ptrs.reserve(parameters_.size());
        for (dl::Tensor& parameter : parameters_) {
            parameter_ptrs.push_back(&parameter);
        }
        optimizer_ = std::make_unique<dl::optim::SGD>(parameter_ptrs, learning_rate);
    }

    void zero_grad() {
        optimizer_->zero_grad();
    }

    void step() {
        optimizer_->step();
    }

private:
    std::vector<dl::Tensor> parameters_;
    std::unique_ptr<dl::optim::SGD> optimizer_;
};

}  // namespace

PYBIND11_MODULE(_C, m) {
    m.doc() = "Python bindings for the dl CUDA deep learning library";

    py::enum_<dl::DeviceType>(m, "DeviceType")
        .value("CPU", dl::DeviceType::CPU)
        .value("CUDA", dl::DeviceType::CUDA);

    py::enum_<dl::DType>(m, "DType")
        .value("Float32", dl::DType::Float32)
        .value("Float16", dl::DType::Float16)
        .value("Int32", dl::DType::Int32)
        .value("Int64", dl::DType::Int64)
        .value("Bool", dl::DType::Bool);

    py::class_<dl::Device>(m, "Device")
        .def(py::init<dl::DeviceType, int>(),
             py::arg("type") = dl::DeviceType::CUDA,
             py::arg("index") = 0)
        .def_readwrite("type", &dl::Device::type)
        .def_readwrite("index", &dl::Device::index)
        .def("is_cuda", &dl::Device::is_cuda)
        .def("is_cpu", &dl::Device::is_cpu)
        .def("__repr__", [](const dl::Device& device) {
            return "dl.Device('" + device.str() + "')";
        });

    py::class_<dl::Shape>(m, "Shape")
        .def(py::init<std::vector<int64_t>>(), py::arg("dims"))
        .def_property_readonly("dims", &shape_dims)
        .def("rank", &dl::Shape::rank)
        .def("numel", &dl::Shape::numel)
        .def("__len__", &dl::Shape::rank)
        .def("__getitem__", [](const dl::Shape& shape, int index) {
            if (index < 0) {
                index += static_cast<int>(shape.rank());
            }
            if (index < 0 || index >= shape.rank()) {
                throw py::index_error("shape index out of range");
            }
            return shape[index];
        })
        .def("__repr__", [](const dl::Shape& shape) {
            return "dl.Shape(" + shape.str() + ")";
        });

    py::class_<dl::Tensor>(m, "Tensor")
        .def(py::init([](const std::vector<int64_t>& dims,
                         dl::DType dtype,
                         dl::Device device) {
                 return dl::Tensor(make_shape(dims), dtype, device);
             }),
             py::arg("shape"),
             py::arg("dtype") = dl::DType::Float32,
             py::arg("device") = dl::Device())
        .def_static("empty", [](const std::vector<int64_t>& dims,
                                dl::DType dtype,
                                dl::Device device) {
            return dl::Tensor::empty(make_shape(dims), dtype, device);
        }, py::arg("shape"), py::arg("dtype") = dl::DType::Float32, py::arg("device") = dl::Device())
        .def_static("zeros", [](const std::vector<int64_t>& dims,
                                dl::DType dtype,
                                dl::Device device) {
            return dl::Tensor::zeros(make_shape(dims), dtype, device);
        }, py::arg("shape"), py::arg("dtype") = dl::DType::Float32, py::arg("device") = dl::Device())
        .def_static("randn", [](const std::vector<int64_t>& dims,
                                dl::DType dtype,
                                dl::Device device,
                                float mean,
                                float stddev,
                                uint64_t seed) {
            return dl::Tensor::randn(make_shape(dims), dtype, device, mean, stddev, seed);
        }, py::arg("shape"), py::arg("dtype") = dl::DType::Float32, py::arg("device") = dl::Device(), py::arg("mean") = 0.0f, py::arg("stddev") = 1.0f, py::arg("seed") = 1234)
        .def_static("uniform", [](const std::vector<int64_t>& dims,
                                  dl::DType dtype,
                                  dl::Device device,
                                  float low,
                                  float high,
                                  uint64_t seed) {
            return dl::Tensor::uniform(make_shape(dims), dtype, device, low, high, seed);
        }, py::arg("shape"), py::arg("dtype") = dl::DType::Float32, py::arg("device") = dl::Device(), py::arg("low") = 0.0f, py::arg("high") = 1.0f, py::arg("seed") = 1234)
        .def_static("from_list", &tensor_from_float_list, py::arg("values"), py::arg("shape"), py::arg("device") = dl::Device())
        .def_static("from_int64_list", &tensor_from_int64_list, py::arg("values"), py::arg("shape"), py::arg("device") = dl::Device())
        .def("tolist", &tensor_to_float_list)
        .def("to_int64_list", &tensor_to_int64_list)
        .def_property_readonly("shape", [](const dl::Tensor& tensor) {
            return tensor.shape();
        })
        .def_property_readonly("dtype", [](const dl::Tensor& tensor) {
            return tensor.dtype();
        })
        .def_property_readonly("device", [](const dl::Tensor& tensor) {
            return tensor.device();
        })
        .def_property("requires_grad", &dl::Tensor::requires_grad, &dl::Tensor::set_requires_grad)
        .def("numel", &dl::Tensor::numel)
        .def("nbytes", &dl::Tensor::nbytes)
        .def("defined", &dl::Tensor::defined)
        .def("zero_", &dl::Tensor::zero_, py::return_value_policy::reference_internal)
        .def("zero_grad", &dl::Tensor::zero_grad)
        .def("grad", &dl::Tensor::grad)
        .def("backward", py::overload_cast<>(&dl::Tensor::backward))
        .def("__add__", [](const dl::Tensor& a, const dl::Tensor& b) { return a + b; }, py::is_operator())
        .def("__sub__", [](const dl::Tensor& a, const dl::Tensor& b) { return a - b; }, py::is_operator())
        .def("__mul__", [](const dl::Tensor& a, const dl::Tensor& b) { return a * b; }, py::is_operator())
        .def("__truediv__", [](const dl::Tensor& a, const dl::Tensor& b) { return a / b; }, py::is_operator())
        .def("__repr__", &tensor_repr);

    m.attr("float32") = dl::DType::Float32;
    m.attr("float16") = dl::DType::Float16;
    m.attr("int32") = dl::DType::Int32;
    m.attr("int64") = dl::DType::Int64;
    m.attr("bool") = dl::DType::Bool;

    m.def("cuda", [](int index) {
        return dl::Device(dl::DeviceType::CUDA, index);
    }, py::arg("index") = 0);
    m.def("cpu", []() {
        return dl::Device(dl::DeviceType::CPU, 0);
    });
    m.def("cuda_synchronize", []() {
        dl::cuda::check(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");
    });
    m.def("is_grad_enabled", &dl::autograd::is_grad_enabled);
    m.def("set_grad_enabled", &dl::autograd::set_grad_enabled, py::arg("enabled"));

    m.def("add", &dl::ops::add);
    m.def("sub", &dl::ops::sub);
    m.def("mul", &dl::ops::mul);
    m.def("div", &dl::ops::div);
    m.def("relu", &dl::ops::relu);
    m.def("leaky_relu", &dl::ops::leaky_relu, py::arg("input"), py::arg("alpha") = 0.01f);
    m.def("gelu", &dl::ops::gelu);
    m.def("sigmoid", &dl::ops::sigmoid);
    m.def("tanh", &dl::ops::tanh);
    m.def("matmul", &dl::ops::matmul);
    m.def("linear", &dl::ops::linear);
    m.def("softmax", &dl::ops::softmax);
    m.def("cross_entropy", &dl::ops::cross_entropy);
    m.def("layernorm", &dl::ops::layernorm, py::arg("input"), py::arg("eps") = 1.0e-5f);
    m.def("layer_norm", &dl::ops::layernorm, py::arg("input"), py::arg("eps") = 1.0e-5f);
    m.def("flatten", &dl::ops::flatten);
    m.def("conv2d", &dl::ops::conv2d, py::arg("input"), py::arg("weight"), py::arg("bias"), py::arg("stride") = 1, py::arg("padding") = 0);
    m.def("max_pool2d", &dl::ops::max_pool2d, py::arg("input"), py::arg("kernel_size"), py::arg("stride") = -1, py::arg("padding") = 0);

    py::class_<dl::nn::Module>(m, "Module")
        .def("forward", &dl::nn::Module::forward)
        .def("__call__", [](dl::nn::Module& module, const dl::Tensor& input) {
            return module.forward(input);
        })
        .def("parameters", &module_parameters)
        .def("num_parameters", &dl::nn::Module::num_parameters)
        .def("train", &dl::nn::Module::train)
        .def("eval", &dl::nn::Module::eval)
        .def("is_training", &dl::nn::Module::is_training);

    py::enum_<dl::nn::LinearInit>(m, "LinearInit")
        .value("XavierUniform", dl::nn::LinearInit::XavierUniform)
        .value("KaimingUniform", dl::nn::LinearInit::KaimingUniform);

    py::class_<dl::nn::Linear, dl::nn::Module>(m, "Linear")
        .def(py::init<int, int, const dl::Device&, dl::nn::LinearInit>(),
             py::arg("in_features"),
             py::arg("out_features"),
             py::arg("device"),
             py::arg("init") = dl::nn::LinearInit::KaimingUniform)
        .def("forward", &dl::nn::Linear::forward)
        .def("__call__", [](dl::nn::Linear& module, const dl::Tensor& input) {
            return module.forward(input);
        })
        .def("parameters", &module_parameters)
        .def("num_parameters", &dl::nn::Linear::num_parameters);

    py::class_<dl::nn::ReLU, dl::nn::Module>(m, "ReLU")
        .def(py::init<>())
        .def("forward", &dl::nn::ReLU::forward)
        .def("__call__", [](dl::nn::ReLU& module, const dl::Tensor& input) {
            return module.forward(input);
        })
        .def("parameters", &module_parameters);

    py::enum_<dl::nn::Conv2DInit>(m, "Conv2DInit")
        .value("XavierUniform", dl::nn::Conv2DInit::XavierUniform)
        .value("KaimingUniform", dl::nn::Conv2DInit::KaimingUniform);

    py::class_<dl::nn::Conv2D, dl::nn::Module>(m, "Conv2D")
        .def(py::init<int, int, int, const dl::Device&, int, int, dl::nn::Conv2DInit>(),
             py::arg("in_channels"),
             py::arg("out_channels"),
             py::arg("kernel_size"),
             py::arg("device"),
             py::arg("stride") = 1,
             py::arg("padding") = 0,
             py::arg("init") = dl::nn::Conv2DInit::KaimingUniform)
        .def("forward", &dl::nn::Conv2D::forward)
        .def("__call__", [](dl::nn::Conv2D& module, const dl::Tensor& input) {
            return module.forward(input);
        })
        .def("parameters", &module_parameters)
        .def("num_parameters", &dl::nn::Conv2D::num_parameters);

    py::class_<dl::nn::MaxPool2D, dl::nn::Module>(m, "MaxPool2D")
        .def(py::init<int, int, int>(),
             py::arg("kernel_size"),
             py::arg("stride") = -1,
             py::arg("padding") = 0)
        .def("forward", &dl::nn::MaxPool2D::forward)
        .def("__call__", [](dl::nn::MaxPool2D& module, const dl::Tensor& input) {
            return module.forward(input);
        })
        .def("parameters", &module_parameters);

    py::class_<dl::nn::Sequential, dl::nn::Module>(m, "Sequential")
        .def(py::init<>())
        .def("add_linear", [](dl::nn::Sequential& seq,
                              int in_features,
                              int out_features,
                              const dl::Device& device,
                              dl::nn::LinearInit init) {
            seq.add(std::make_unique<dl::nn::Linear>(
                in_features,
                out_features,
                device,
                init));
            return &seq;
        }, py::arg("in_features"), py::arg("out_features"), py::arg("device"), py::arg("init") = dl::nn::LinearInit::KaimingUniform, py::return_value_policy::reference)
        .def("add_relu", [](dl::nn::Sequential& seq) {
            seq.add(std::make_unique<dl::nn::ReLU>());
            return &seq;
        }, py::return_value_policy::reference)
        .def("add_conv2d", [](dl::nn::Sequential& seq,
                              int in_channels,
                              int out_channels,
                              int kernel_size,
                              const dl::Device& device,
                              int stride,
                              int padding,
                              dl::nn::Conv2DInit init) {
            seq.add(std::make_unique<dl::nn::Conv2D>(
                in_channels,
                out_channels,
                kernel_size,
                device,
                stride,
                padding,
                init));
            return &seq;
        }, py::arg("in_channels"), py::arg("out_channels"), py::arg("kernel_size"), py::arg("device"), py::arg("stride") = 1, py::arg("padding") = 0, py::arg("init") = dl::nn::Conv2DInit::KaimingUniform, py::return_value_policy::reference)
        .def("add_max_pool2d", [](dl::nn::Sequential& seq,
                                  int kernel_size,
                                  int stride,
                                  int padding) {
            seq.add(std::make_unique<dl::nn::MaxPool2D>(
                kernel_size,
                stride,
                padding));
            return &seq;
        }, py::arg("kernel_size"), py::arg("stride") = -1, py::arg("padding") = 0, py::return_value_policy::reference)
        .def("forward", &dl::nn::Sequential::forward)
        .def("__call__", [](dl::nn::Sequential& module, const dl::Tensor& input) {
            return module.forward(input);
        })
        .def("parameters", &module_parameters)
        .def("num_parameters", &dl::nn::Sequential::num_parameters);

    py::class_<PySGD>(m, "SGD")
        .def(py::init<std::vector<dl::Tensor>, float>(),
             py::arg("parameters"),
             py::arg("learning_rate"))
        .def("zero_grad", &PySGD::zero_grad)
        .def("step", &PySGD::step);
}
