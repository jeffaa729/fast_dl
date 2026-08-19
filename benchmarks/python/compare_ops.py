import argparse
import csv
from pathlib import Path


ORDER = [
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


def load(path: Path):
    if not path.exists():
        return {}
    with path.open(newline="") as f:
        return {row["op"]: row for row in csv.DictReader(f)}


def avg_ms(rows, op):
    if op not in rows:
        return None
    return float(rows[op]["avg_ms"])


def fmt(value):
    return "n/a" if value is None else f"{value:.4f}"


def speedup(reference, candidate):
    if reference is None or candidate is None or candidate == 0:
        return "n/a"
    return f"{reference / candidate:.2f}x"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpp", default="benchmark_results/ops.csv")
    parser.add_argument("--dl-python", default="benchmark_results/dl_python_ops.csv")
    parser.add_argument("--pytorch", default="benchmark_results/pytorch_python_ops.csv")
    parser.add_argument("--output", default="benchmark_results/compare_python_ops.csv")
    args = parser.parse_args()

    cpp = load(Path(args.cpp))
    dl_python = load(Path(args.dl_python))
    pytorch = load(Path(args.pytorch))

    rows = []
    for op in ORDER:
        shape = (
            cpp.get(op, {}).get("shape")
            or dl_python.get(op, {}).get("shape")
            or pytorch.get(op, {}).get("shape")
            or "n/a"
        )
        cpp_ms = avg_ms(cpp, op)
        dl_py_ms = avg_ms(dl_python, op)
        torch_ms = avg_ms(pytorch, op)
        rows.append({
            "op": op,
            "shape": shape,
            "cpp_dl_ms": cpp_ms,
            "python_dl_ms": dl_py_ms,
            "pytorch_ms": torch_ms,
            "cpp_vs_pytorch": speedup(torch_ms, cpp_ms),
            "python_overhead": speedup(dl_py_ms, cpp_ms),
            "python_dl_vs_pytorch": speedup(torch_ms, dl_py_ms),
        })

    print("compare : cpp dl vs python dl vs pytorch")
    print(
        f"{'op':<16} {'shape':<24} {'cpp_dl':>10} {'py_dl':>10} "
        f"{'pytorch':>10} {'cpp/torch':>11} {'py/cpp':>9} {'py/torch':>10}"
    )
    print("-" * 106)
    for row in rows:
        print(
            f"{row['op']:<16} {row['shape']:<24} {fmt(row['cpp_dl_ms']):>10} "
            f"{fmt(row['python_dl_ms']):>10} {fmt(row['pytorch_ms']):>10} "
            f"{row['cpp_vs_pytorch']:>11} {row['python_overhead']:>9} "
            f"{row['python_dl_vs_pytorch']:>10}"
        )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output).open("w", newline="") as f:
        fieldnames = [
            "op",
            "shape",
            "cpp_dl_ms",
            "python_dl_ms",
            "pytorch_ms",
            "cpp_vs_pytorch",
            "python_overhead",
            "python_dl_vs_pytorch",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"compare_python_ops_csv : {args.output}")


if __name__ == "__main__":
    main()
