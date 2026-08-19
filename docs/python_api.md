# Python API

The Python package is a thin pybind11 wrapper over the C++/CUDA core. It exposes the user-facing API only: tensors, devices, dtypes, common ops, basic modules, and SGD.

## Build

```bash
CUDACXX=/usr/local/cuda-13.3/bin/nvcc uv sync --extra test --extra benchmark
```

## Test

```bash
uv run pytest tests/python
```

The C++ tests should still remain, because they test the core library directly. The Python tests only check the user-facing binding layer.

## Example

```python
import dl

x = dl.Tensor.randn((2, 3), dl.float32, dl.cuda(0))
y = dl.relu(x)
print(y.shape.dims)
```

`examples/minimal.py` is the quick smoke test. `examples/mnist_mlp.py` and `examples/cifar10_cnn.py` are full train/eval demos that load data, train for fixed epochs, print loss/accuracy, and show sample predictions.

Prefer `Tensor.from_numpy(array, device)` for data loading. It copies directly from a contiguous NumPy `float32` or `int64` array into a CUDA tensor and avoids slow Python list construction.

## Benchmark

```bash
bash scripts/compare_python_ops.sh
```

The comparison has three meanings:

- C++ dl vs PyTorch: kernel/backend quality.
- Python dl vs C++ dl: pybind and Python overhead.
- Python dl vs PyTorch: user-facing API performance.

Use `dl.no_grad()` for inference-style timing so autograd graph construction does not pollute op latency.
