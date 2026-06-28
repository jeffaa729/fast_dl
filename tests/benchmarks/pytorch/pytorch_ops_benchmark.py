#!/usr/bin/env python3

import csv
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List

try:
    import torch
    import torch.nn.functional as F
except ImportError as error:
    print(f"pytorch_ops_benchmark_error : {error}", file=sys.stderr)
    sys.exit(1)


@dataclass
class Result:
    op: str
    shape: str
    iterations: int
    warmup: int
    total_ms: float
    avg_ms: float


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def env_string(name: str, default: str) -> str:
    return os.environ.get(name, default)


def time_cuda_ms(warmup: int, iterations: int, fn: Callable[[], torch.Tensor]) -> float:
    with torch.no_grad():
        for _ in range(warmup):
            y = fn()
            _ = y.data_ptr()

        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iterations):
            y = fn()
            _ = y.data_ptr()
        torch.cuda.synchronize()
        end = time.perf_counter()

    return (end - start) * 1000.0


def make_result(
    op: str, shape: str, warmup: int, iterations: int, total_ms: float
) -> Result:
    return Result(
        op=op,
        shape=shape,
        iterations=iterations,
        warmup=warmup,
        total_ms=total_ms,
        avg_ms=total_ms / float(iterations),
    )


def benchmark_matmul(device: torch.device, warmup: int, iterations: int) -> Result:
    m, n, k = 1024, 1024, 1024
    generator = torch.Generator(device=device).manual_seed(100)
    a = torch.randn((m, k), device=device, dtype=torch.float32, generator=generator)
    b = torch.randn((k, n), device=device, dtype=torch.float32, generator=generator)
    total_ms = time_cuda_ms(warmup, iterations, lambda: torch.matmul(a, b))
    return make_result("matmul", "1024x1024x1024", warmup, iterations, total_ms)


def benchmark_linear(device: torch.device, warmup: int, iterations: int) -> Result:
    batch, in_features, out_features = 1024, 784, 128
    generator = torch.Generator(device=device).manual_seed(200)
    x = torch.randn(
        (batch, in_features), device=device, dtype=torch.float32, generator=generator
    )
    weight = torch.randn(
        (out_features, in_features),
        device=device,
        dtype=torch.float32,
        generator=generator,
    )
    bias = torch.zeros((out_features,), device=device, dtype=torch.float32)
    total_ms = time_cuda_ms(warmup, iterations, lambda: F.linear(x, weight, bias))
    return make_result("linear", "1024x784x128", warmup, iterations, total_ms)


def benchmark_unary(
    op: str,
    device: torch.device,
    warmup: int,
    iterations: int,
    seed: int,
    fn: Callable[[torch.Tensor], torch.Tensor],
) -> Result:
    elements = 1 << 20
    generator = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn((elements,), device=device, dtype=torch.float32, generator=generator)
    total_ms = time_cuda_ms(warmup, iterations, lambda: fn(x))
    return make_result(op, "1048576", warmup, iterations, total_ms)


def benchmark_binary(
    op: str,
    device: torch.device,
    warmup: int,
    iterations: int,
    seed: int,
    fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> Result:
    elements = 1 << 20
    generator = torch.Generator(device=device).manual_seed(seed)
    a = torch.randn((elements,), device=device, dtype=torch.float32, generator=generator)
    b = torch.randn((elements,), device=device, dtype=torch.float32, generator=generator)
    total_ms = time_cuda_ms(warmup, iterations, lambda: fn(a, b))
    return make_result(op, "1048576", warmup, iterations, total_ms)


def benchmark_softmax(device: torch.device, warmup: int, iterations: int) -> Result:
    rows, cols = 4096, 1024
    generator = torch.Generator(device=device).manual_seed(400)
    x = torch.randn((rows, cols), device=device, dtype=torch.float32, generator=generator)
    total_ms = time_cuda_ms(warmup, iterations, lambda: F.softmax(x, dim=1))
    return make_result("softmax", "4096x1024", warmup, iterations, total_ms)


def benchmark_layernorm(device: torch.device, warmup: int, iterations: int) -> Result:
    rows, cols = 4096, 1024
    generator = torch.Generator(device=device).manual_seed(500)
    x = torch.randn((rows, cols), device=device, dtype=torch.float32, generator=generator)
    total_ms = time_cuda_ms(warmup, iterations, lambda: F.layer_norm(x, (cols,)))
    return make_result("layernorm", "4096x1024", warmup, iterations, total_ms)


def benchmark_cross_entropy(
    device: torch.device, warmup: int, iterations: int
) -> Result:
    batch, classes = 4096, 1000
    generator = torch.Generator(device=device).manual_seed(600)
    logits = torch.randn(
        (batch, classes), device=device, dtype=torch.float32, generator=generator
    )
    labels = torch.arange(batch, device=device, dtype=torch.long) % classes
    total_ms = time_cuda_ms(
        warmup, iterations, lambda: F.cross_entropy(logits, labels)
    )
    return make_result("cross_entropy", "4096x1000", warmup, iterations, total_ms)


def write_csv(path: str, results: List[Result]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["op", "shape", "iterations", "warmup", "total_ms", "avg_ms"])
        for result in results:
            writer.writerow(
                [
                    result.op,
                    result.shape,
                    result.iterations,
                    result.warmup,
                    result.total_ms,
                    result.avg_ms,
                ]
            )


def print_results(results: List[Result]) -> None:
    print("pytorch_ops_benchmark : results")
    print(
        f"{'op':<18}{'shape':<20}{'iters':>12}{'warmup':>10}"
        f"{'total_ms':>14}{'avg_ms':>12}"
    )
    print("-" * 86)
    for result in results:
        print(
            f"{result.op:<18}{result.shape:<20}"
            f"{result.iterations:>12}{result.warmup:>10}"
            f"{result.total_ms:>14.4f}{result.avg_ms:>12.4f}"
        )


def main() -> int:
    if not torch.cuda.is_available():
        print("pytorch_ops_benchmark_error : CUDA is not available", file=sys.stderr)
        return 1

    warmup = env_int("OPS_BENCH_WARMUP", 10)
    iterations = env_int("OPS_BENCH_ITERS", 100)
    csv_path = env_string("PYTORCH_BENCH_CSV", "benchmark_results/pytorch_ops.csv")
    device = torch.device("cuda:0")

    torch.manual_seed(1234)
    torch.cuda.set_device(device)

    results = [
        benchmark_binary("add", device, warmup, iterations, 10, torch.add),
        benchmark_binary("sub", device, warmup, iterations, 20, torch.sub),
        benchmark_binary("mul", device, warmup, iterations, 30, torch.mul),
        benchmark_binary("div", device, warmup, iterations, 40, torch.div),
        benchmark_unary("relu", device, warmup, iterations, 300, torch.relu),
        benchmark_unary(
            "leaky_relu",
            device,
            warmup,
            iterations,
            310,
            lambda x: F.leaky_relu(x, negative_slope=0.01),
        ),
        benchmark_unary("gelu", device, warmup, iterations, 320, F.gelu),
        benchmark_unary("sigmoid", device, warmup, iterations, 330, torch.sigmoid),
        benchmark_unary("tanh", device, warmup, iterations, 340, torch.tanh),
        benchmark_matmul(device, warmup, iterations),
        benchmark_linear(device, warmup, iterations),
        benchmark_softmax(device, warmup, iterations),
        benchmark_layernorm(device, warmup, iterations),
        benchmark_cross_entropy(device, warmup, iterations),
    ]

    write_csv(csv_path, results)
    print_results(results)
    print(f"pytorch_ops_benchmark_csv : {csv_path}")
    print("pytorch_ops_benchmark : passed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"pytorch_ops_benchmark_error : {error}", file=sys.stderr)
        print("pytorch_ops_benchmark : not passed")
        raise SystemExit(1)
