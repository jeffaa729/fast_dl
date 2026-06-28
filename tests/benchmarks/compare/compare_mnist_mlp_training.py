#!/usr/bin/env python3

import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Row:
    name: str
    epochs: int
    batch_size: int
    train_batches: int
    total_ms: float
    avg_epoch_ms: float


def env_string(name: str, default: str) -> str:
    return os.environ.get(name, default)


def read_one(path: str) -> Row:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"missing CSV: {path}")

    with csv_path.open(newline="") as file:
        reader = csv.DictReader(file)
        required = {
            "name",
            "epochs",
            "batch_size",
            "train_batches",
            "total_ms",
            "avg_epoch_ms",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise RuntimeError(f"CSV has wrong header: {path}")
        rows = list(reader)

    if len(rows) != 1:
        raise RuntimeError(f"CSV must contain exactly one result row: {path}")

    item = rows[0]
    return Row(
        name=item["name"],
        epochs=int(item["epochs"]),
        batch_size=int(item["batch_size"]),
        train_batches=int(item["train_batches"]),
        total_ms=float(item["total_ms"]),
        avg_epoch_ms=float(item["avg_epoch_ms"]),
    )


def write_csv(path: str, dl_row: Row, torch_row: Row, speedup: float) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "benchmark",
            "epochs",
            "batch_size",
            "dl_total_ms",
            "pytorch_total_ms",
            "dl_avg_epoch_ms",
            "pytorch_avg_epoch_ms",
            "speedup",
        ])
        writer.writerow([
            "mnist_mlp_training",
            dl_row.epochs,
            dl_row.batch_size,
            dl_row.total_ms,
            torch_row.total_ms,
            dl_row.avg_epoch_ms,
            torch_row.avg_epoch_ms,
            speedup,
        ])


def main() -> int:
    dl_csv = env_string(
        "DL_MNIST_MLP_CSV",
        "benchmark_results/dl_mnist_mlp_training.csv",
    )
    torch_csv = env_string(
        "PYTORCH_MNIST_MLP_CSV",
        "benchmark_results/pytorch_mnist_mlp_training.csv",
    )
    compare_csv = env_string(
        "COMPARE_MNIST_MLP_CSV",
        "benchmark_results/compare_mnist_mlp_training.csv",
    )

    dl_row = read_one(dl_csv)
    torch_row = read_one(torch_csv)
    speedup = torch_row.total_ms / dl_row.total_ms if dl_row.total_ms > 0.0 else float("inf")

    print("compare_mnist_mlp_training : results")
    print(
        f"{'benchmark':<24}{'epochs':>8}{'batch':>8}"
        f"{'dl_ms':>14}{'pytorch_ms':>14}{'speedup':>10}"
    )
    print("-" * 78)
    print(
        f"{'mnist_mlp_training':<24}{dl_row.epochs:>8}{dl_row.batch_size:>8}"
        f"{dl_row.total_ms:>14.4f}{torch_row.total_ms:>14.4f}{speedup:>10.2f}x"
    )

    write_csv(compare_csv, dl_row, torch_row, speedup)
    print(f"compare_mnist_mlp_csv : {compare_csv}")
    print("compare_mnist_mlp_training : passed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"compare_mnist_mlp_training_error : {error}", file=sys.stderr)
        print("compare_mnist_mlp_training : not passed")
        raise SystemExit(1)
