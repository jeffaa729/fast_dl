import pytest

import dl


def test_data_api_status():
    assert hasattr(dl, "Tensor")
