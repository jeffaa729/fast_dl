#!/usr/bin/env python3

import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class Row:
    op: str
    shape: str
    iterations: int
    warmup: int
    total_ms: float
    avg_ms: float


def env_string(name: str, default: str) -> str:
    return os.environ.get(name, default)


def read_csv(path: str) -> List[Row]:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"missing CSV: {path}")

    rows: List[Row] = []
    with csv_path.open(newline="") as file:
        reader = csv.DictReader(file)
        required = {"op", "shape", "iterations", "warmup", "total_ms", "avg_ms"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise RuntimeError(f"CSV has wrong header: {path}")

        for item in reader:
            row = Row(
                op=item["op"],
                shape=item["shape"],
                iterations=int(item["iterations"]),
                warmup=int(item["warmup"]),
                total_ms=float(item["total_ms"]),
                avg_ms=float(item["avg_ms"]),
            )
            rows.append(row)

    return rows


def compare(dl_rows: List[Row],
            torch_rows: List[Row]) -> List[Tuple[Row, Row, float]]:
    torch_by_key: Dict[Tuple[str, str], Row] = {
        (row.op, row.shape): row for row in torch_rows
    }
    results: List[Tuple[Row, Row, float]] = []
    for dl_row in dl_rows:
        key = (dl_row.op, dl_row.shape)
        if key not in torch_by_key:
            continue
        torch_row = torch_by_key[key]
        speedup = torch_row.avg_ms / dl_row.avg_ms if dl_row.avg_ms > 0.0 else float("inf")
        results.append((dl_row, torch_row, speedup))
    return results


def write_csv(path: str, rows: List[Tuple[Row, Row, float]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "op",
            "shape",
            "dl_avg_ms",
            "pytorch_avg_ms",
            "speedup",
            "dl_total_ms",
            "pytorch_total_ms",
            "iterations",
            "warmup",
        ])
        for dl_row, torch_row, speedup in rows:
            writer.writerow([
                dl_row.op,
                dl_row.shape,
                dl_row.avg_ms,
                torch_row.avg_ms,
                speedup,
                dl_row.total_ms,
                torch_row.total_ms,
                dl_row.iterations,
                dl_row.warmup,
            ])


def print_table(rows: List[Tuple[Row, Row, float]]) -> None:
    print("compare_ops : results")
    print(
        f"{'op':<18}{'shape':<20}"
        f"{'dl_ms':>12}{'pytorch_ms':>14}{'speedup':>10}"
    )
    print("-" * 74)
    for dl_row, torch_row, speedup in rows:
        print(
            f"{dl_row.op:<18}{dl_row.shape:<20}"
            f"{dl_row.avg_ms:>12.4f}{torch_row.avg_ms:>14.4f}{speedup:>10.2f}x"
        )


def main() -> int:
    dl_csv = env_string("OPS_BENCH_CSV", "benchmark_results/ops.csv")
    torch_csv = env_string("PYTORCH_BENCH_CSV", "benchmark_results/pytorch_ops.csv")
    compare_csv = env_string("COMPARE_BENCH_CSV", "benchmark_results/compare_ops.csv")

    dl_rows = read_csv(dl_csv)
    torch_rows = read_csv(torch_csv)
    rows = compare(dl_rows, torch_rows)

    if not rows:
        raise RuntimeError("no matching op/shape rows to compare")

    print_table(rows)
    write_csv(compare_csv, rows)
    print(f"compare_ops_csv : {compare_csv}")
    print("compare_ops : passed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"compare_ops_error : {error}", file=sys.stderr)
        print("compare_ops : not passed")
        raise SystemExit(1)
