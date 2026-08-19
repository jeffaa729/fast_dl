# Python API

The Python package is a thin pybind11 wrapper over the C++/CUDA core. It exposes the user-facing API only: tensors, devices, dtypes, common ops, basic modules, and SGD.

## Build

```bash
python3 -m venv python-env
source python-env/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test,benchmark]"
```

## Test

```bash
python -m pytest tests/python
```

The C++ tests should still remain, because they test the core library directly. The Python tests only check the user-facing binding layer.

## Example

```python
import dl

x = dl.Tensor.randn((2, 3), dl.float32, dl.cuda(0))
y = dl.relu(x)
print(y.shape.dims)
```

## Benchmark

```bash
bash scripts/compare_python_ops.sh
```

The comparison has three meanings:

- C++ dl vs PyTorch: kernel/backend quality.
- Python dl vs C++ dl: pybind and Python overhead.
- Python dl vs PyTorch: user-facing API performance.

Use `dl.no_grad()` for inference-style timing so autograd graph construction does not pollute op latency.
