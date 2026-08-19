import argparse
import csv
import os
import time

import dl


OPS = [
    "add",
    "sub",
    "mul",
    "div",
    "relu",
    "leaky_relu",
    "gelu",
    "sigmoid",
    "tanh",
    "matmul",
    "linear",
    "conv2d_3x3",
    "softmax",
    "layernorm",
    "cross_entropy",
    "max_pool2d",
]


def synchronize() -> None:
    dl.cuda_synchronize()


def bench(name: str, iterations: int, warmup: int):
    device = dl.cuda(0)

    if name in {"add", "sub", "mul", "div"}:
        shape = "1048576"
        a = dl.Tensor.randn((1048576,), dl.float32, device)
        b = dl.Tensor.randn((1048576,), dl.float32, device)
        fn = {
            "add": lambda: dl.add(a, b),
            "sub": lambda: dl.sub(a, b),
            "mul": lambda: dl.mul(a, b),
            "div": lambda: dl.div(a, b),
        }[name]
    elif name in {"relu", "leaky_relu", "gelu", "sigmoid", "tanh"}:
        shape = "1048576"
        x = dl.Tensor.randn((1048576,), dl.float32, device)
        fn = {
            "relu": lambda: dl.relu(x),
            "leaky_relu": lambda: dl.leaky_relu(x),
            "gelu": lambda: dl.gelu(x),
            "sigmoid": lambda: dl.sigmoid(x),
            "tanh": lambda: dl.tanh(x),
        }[name]
    elif name == "matmul":
        shape = "1024x1024x1024"
        a = dl.Tensor.randn((1024, 1024), dl.float32, device)
        b = dl.Tensor.randn((1024, 1024), dl.float32, device)
        fn = lambda: dl.matmul(a, b)
    elif name == "linear":
        shape = "1024x784x128"
        x = dl.Tensor.randn((1024, 784), dl.float32, device)
        weight = dl.Tensor.randn((784, 128), dl.float32, device)
        bias = dl.Tensor.zeros((128,), dl.float32, device)
        fn = lambda: dl.linear(x, weight, bias)
    elif name == "conv2d_3x3":
        shape = "64x16x16x16->32"
        x = dl.Tensor.randn((64, 16, 16, 16), dl.float32, device)
        weight = dl.Tensor.randn((32, 16, 3, 3), dl.float32, device)
        bias = dl.Tensor.zeros((32,), dl.float32, device)
        fn = lambda: dl.conv2d(x, weight, bias, stride=1, padding=1)
    elif name == "softmax":
        shape = "4096x1024"
        x = dl.Tensor.randn((4096, 1024), dl.float32, device)
        fn = lambda: dl.softmax(x)
    elif name == "layernorm":
        shape = "4096x1024"
        x = dl.Tensor.randn((4096, 1024), dl.float32, device)
        fn = lambda: dl.layernorm(x)
    elif name == "cross_entropy":
        shape = "4096x1000"
        logits = dl.Tensor.randn((4096, 1000), dl.float32, device)
        labels = dl.Tensor.from_int64_list([i % 1000 for i in range(4096)], (4096,), device)
        fn = lambda: dl.cross_entropy(logits, labels)
    elif name == "max_pool2d":
        shape = "64x32x32x32,k=2"
        x = dl.Tensor.randn((64, 32, 32, 32), dl.float32, device)
        fn = lambda: dl.max_pool2d(x, 2)
    else:
        raise ValueError(f"unknown op: {name}")

    for _ in range(warmup):
        fn()
    synchronize()

    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    synchronize()
    total_ms = (time.perf_counter() - start) * 1000.0

    return {
        "op": name,
        "shape": shape,
        "iterations": iterations,
        "warmup": warmup,
        "total_ms": total_ms,
        "avg_ms": total_ms / iterations,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="benchmark_results/dl_python_ops.csv")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with dl.no_grad():
        rows = [bench(op, args.iterations, args.warmup) for op in OPS]

    print("benchmark : dl python ops")
    print(f"{'op':<16} {'shape':<24} {'iters':>8} {'warmup':>8} {'total_ms':>12} {'avg_ms':>11}")
    print("-" * 83)
    for row in rows:
        print(
            f"{row['op']:<16} {row['shape']:<24} {row['iterations']:>8} "
            f"{row['warmup']:>8} {row['total_ms']:>12.4f} {row['avg_ms']:>11.4f}"
        )

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["op", "shape", "iterations", "warmup", "total_ms", "avg_ms"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"dl_python_ops_benchmark_csv : {args.output}")


if __name__ == "__main__":
    main()
