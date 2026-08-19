import pytest

import dl


def test_elementwise_ops():
    device = dl.cuda(0)
    a = dl.Tensor.from_list([2.0, 4.0, 6.0], (3,), device)
    b = dl.Tensor.from_list([1.0, 2.0, 3.0], (3,), device)

    assert dl.add(a, b).tolist() == pytest.approx([3.0, 6.0, 9.0])
    assert (a - b).tolist() == pytest.approx([1.0, 2.0, 3.0])
    assert dl.mul(a, b).tolist() == pytest.approx([2.0, 8.0, 18.0])
    assert (a / b).tolist() == pytest.approx([2.0, 2.0, 2.0])


def test_relu():
    x = dl.Tensor.from_list([-2.0, 0.0, 3.0], (3,), dl.cuda(0))

    assert dl.relu(x).tolist() == pytest.approx([0.0, 0.0, 3.0])


def test_matmul():
    device = dl.cuda(0)
    a = dl.Tensor.from_list([1.0, 2.0, 3.0, 4.0], (2, 2), device)
    b = dl.Tensor.from_list([5.0, 6.0, 7.0, 8.0], (2, 2), device)

    assert dl.matmul(a, b).tolist() == pytest.approx([19.0, 22.0, 43.0, 50.0])


def test_softmax_rows_sum_to_one():
    x = dl.Tensor.from_list([1.0, 2.0, 3.0, 1.0, 1.0, 1.0], (2, 3), dl.cuda(0))
    y = dl.softmax(x).tolist()

    assert sum(y[:3]) == pytest.approx(1.0, abs=1.0e-5)
    assert sum(y[3:]) == pytest.approx(1.0, abs=1.0e-5)
