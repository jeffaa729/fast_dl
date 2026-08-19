import pytest

import dl


def cuda_device():
    return dl.cuda(0)


def test_tensor_numel_nbytes():
    x = dl.Tensor((2, 3), dl.float32, cuda_device())

    assert x.numel() == 6
    assert x.nbytes() == 24
    assert x.shape.dims == [2, 3]
    assert x.dtype == dl.float32
    assert x.device.is_cuda()


def test_tensor_from_list_to_list():
    x = dl.Tensor.from_list([1.0, 2.0, 3.0, 4.0], (2, 2), cuda_device())

    assert x.tolist() == pytest.approx([1.0, 2.0, 3.0, 4.0])


def test_tensor_zeros():
    x = dl.Tensor.zeros((4,), dl.float32, cuda_device())

    assert x.tolist() == pytest.approx([0.0, 0.0, 0.0, 0.0])
