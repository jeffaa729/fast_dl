import pytest

import dl


def test_add_backward():
    device = dl.cuda(0)
    x = dl.Tensor.from_list([1.0, 2.0, 3.0], (3,), device)
    y = dl.Tensor.from_list([4.0, 5.0, 6.0], (3,), device)
    x.requires_grad = True
    y.requires_grad = True

    z = x + y
    z.backward()

    assert x.grad().tolist() == pytest.approx([1.0, 1.0, 1.0])
    assert y.grad().tolist() == pytest.approx([1.0, 1.0, 1.0])


def test_no_grad_context_restores_state():
    assert dl.is_grad_enabled()
    with dl.no_grad():
        assert not dl.is_grad_enabled()
    assert dl.is_grad_enabled()
