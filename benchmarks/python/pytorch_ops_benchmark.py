import argparse
import csv
import os
import time

import torch
import torch.nn.functional as F


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


def bench(name: str, iterations: int, warmup: int):
    device = "cuda"

    if name in {"add", "sub", "mul", "div"}:
        shape = "1048576"
        a = torch.randn(1048576, device=device)
        b = torch.randn(1048576, device=device)
        fn = {
            "add": lambda: a + b,
            "sub": lambda: a - b,
            "mul": lambda: a * b,
            "div": lambda: a / b,
        }[name]
    elif name in {"relu", "leaky_relu", "gelu", "sigmoid", "tanh"}:
        shape = "1048576"
        x = torch.randn(1048576, device=device)
        fn = {
            "relu": lambda: F.relu(x),
            "leaky_relu": lambda: F.leaky_relu(x, negative_slope=0.01),
            "gelu": lambda: F.gelu(x),
            "sigmoid": lambda: torch.sigmoid(x),
            "tanh": lambda: torch.tanh(x),
        }[name]
    elif name == "matmul":
        shape = "1024x1024x1024"
        a = torch.randn(1024, 1024, device=device)
        b = torch.randn(1024, 1024, device=device)
        fn = lambda: a @ b
    elif name == "linear":
        shape = "1024x784x128"
        x = torch.randn(1024, 784, device=device)
        weight = torch.randn(128, 784, device=device)
        bias = torch.zeros(128, device=device)
        fn = lambda: F.linear(x, weight, bias)
    elif name == "conv2d_3x3":
        shape = "64x16x16x16->32"
        x = torch.randn(64, 16, 16, 16, device=device)
        weight = torch.randn(32, 16, 3, 3, device=device)
        bias = torch.zeros(32, device=device)
        fn = lambda: F.conv2d(x, weight, bias, stride=1, padding=1)
    elif name == "softmax":
        shape = "4096x1024"
        x = torch.randn(4096, 1024, device=device)
        fn = lambda: F.softmax(x, dim=1)
    elif name == "layernorm":
        shape = "4096x1024"
        x = torch.randn(4096, 1024, device=device)
        fn = lambda: F.layer_norm(x, (1024,))
    elif name == "cross_entropy":
        shape = "4096x1000"
        logits = torch.randn(4096, 1000, device=device)
        labels = torch.tensor([i % 1000 for i in range(4096)], dtype=torch.long, device=device)
        fn = lambda: F.cross_entropy(logits, labels)
    elif name == "max_pool2d":
        shape = "64x32x32x32,k=2"
        x = torch.randn(64, 32, 32, 32, device=device)
        fn = lambda: F.max_pool2d(x, 2)
    else:
        raise ValueError(f"unknown op: {name}")

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    torch.cuda.synchronize()
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
    parser.add_argument("--output", default="benchmark_results/pytorch_python_ops.csv")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    rows = [bench(op, args.iterations, args.warmup) for op in OPS]

    print("benchmark : pytorch python ops")
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
    print(f"pytorch_python_ops_benchmark_csv : {args.output}")


if __name__ == "__main__":
    main()
