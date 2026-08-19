import pytest

import dl


def test_linear_module_forward_shape_and_params():
    device = dl.cuda(0)
    layer = dl.Linear(3, 2, device)
    x = dl.Tensor.randn((4, 3), dl.float32, device)

    y = layer(x)

    assert y.shape.dims == [4, 2]
    assert layer.num_parameters() == 8


def test_relu_module_forward():
    relu = dl.ReLU()
    x = dl.Tensor.from_list([-1.0, 2.0], (2,), dl.cuda(0))

    assert relu(x).tolist() == pytest.approx([0.0, 2.0])


def test_sequential_forward():
    device = dl.cuda(0)
    model = dl.Sequential()
    model.add_linear(3, 4, device).add_relu().add_linear(4, 2, device)
    x = dl.Tensor.randn((5, 3), dl.float32, device)

    y = model(x)

    assert y.shape.dims == [5, 2]
    assert model.num_parameters() == 26


def test_sgd_accepts_module_parameters():
    device = dl.cuda(0)
    layer = dl.Linear(3, 2, device)
    opt = dl.SGD(layer.parameters(), 0.01)

    opt.zero_grad()
