#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${BUILD_DIR:-build}"
REPORT_DIR="${REPORT_DIR:-ncu_reports}"
CUDA_ARCH="${CUDA_ARCH:-75}"
# vector_add is disabled by default for now. Run it explicitly with:
# BENCHMARKS="vector_add" bash scripts/run_linux_benchmarks.sh
BENCHMARKS="${BENCHMARKS:-transpose reduction gemm softmax}"
NCU_METRICS="${NCU_METRICS:-gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,lts__throughput.avg.pct_of_peak_sustained_elapsed,l1tex__throughput.avg.pct_of_peak_sustained_elapsed}"
CUDA_DRIVER_SHIM_DIR="${CUDA_DRIVER_SHIM_DIR:-.cuda-driver-lib}"

clean_ld_library_path() {
    if [[ -z "${LD_LIBRARY_PATH:-}" ]]; then
        return
    fi

    local cleaned=""
    local entry
    IFS=':' read -ra entries <<< "${LD_LIBRARY_PATH}"
    for entry in "${entries[@]}"; do
        if [[ "${entry}" == *"/stubs"* ]]; then
            continue
        fi
        if [[ -z "${cleaned}" ]]; then
            cleaned="${entry}"
        else
            cleaned="${cleaned}:${entry}"
        fi
    done
    export LD_LIBRARY_PATH="${cleaned}"
}

prefer_wsl_cuda_driver() {
    local wsl_driver="/usr/lib/wsl/lib/libcuda.so.1"

    if [[ ! -e "${wsl_driver}" ]]; then
        return
    fi

    mkdir -p "${CUDA_DRIVER_SHIM_DIR}"
    ln -sfn "${wsl_driver}" "${CUDA_DRIVER_SHIM_DIR}/libcuda.so"
    ln -sfn "${wsl_driver}" "${CUDA_DRIVER_SHIM_DIR}/libcuda.so.1"

    if [[ -z "${LD_LIBRARY_PATH:-}" ]]; then
        export LD_LIBRARY_PATH="${PWD}/${CUDA_DRIVER_SHIM_DIR}:/usr/lib/wsl/lib"
    else
        export LD_LIBRARY_PATH="${PWD}/${CUDA_DRIVER_SHIM_DIR}:/usr/lib/wsl/lib:${LD_LIBRARY_PATH}"
    fi
}

print_cuda_driver_diagnostics() {
    echo "ncu_path : $(command -v ncu)"
    echo "ld_library_path : ${LD_LIBRARY_PATH:-<empty>}"

    if command -v ldconfig >/dev/null 2>&1; then
        echo "libcuda_candidates :"
        ldconfig -p 2>/dev/null | grep 'libcuda.so' || true
    fi

    if command -v nvidia-smi >/dev/null 2>&1; then
        echo "nvidia_smi :"
        nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true
    else
        echo "nvidia_smi : not found"
    fi
}

profile_benchmark() {
    local name="$1"
    local report="${REPORT_DIR}/${name}"
    local csv="${REPORT_DIR}/${name}_metrics.csv"
    local log="${REPORT_DIR}/${name}_ncu.log"

    case "$name" in
        vector_add)
            local args=(vector_add "${VECTOR_ADD_SIZE:-16777216}")
            ;;
        transpose)
            local args=(transpose "${TRANSPOSE_N:-4096}")
            ;;
        reduction)
            local args=(reduction "${REDUCTION_SIZE:-16777216}")
            ;;
        gemm)
            local args=(gemm "${GEMM_N:-512}")
            ;;
        softmax)
            local args=(softmax "${SOFTMAX_ROWS:-4096}" "${SOFTMAX_COLS:-1024}")
            ;;
        *)
            echo "unknown benchmark: ${name}" >&2
            return 1
            ;;
    esac

    echo
    echo "ncu_profile : ${name}"
    if ! ncu --force-overwrite \
        --metrics "${NCU_METRICS}" \
        -o "${report}" \
        "${BUILD_DIR}/dl_benchmarks" "${args[@]}" \
        > "${log}" 2>&1; then
        echo "ncu_profile : ${name} report failed" >&2
        cat "${log}" >&2
        return 1
    fi

    if ! ncu --metrics "${NCU_METRICS}" \
        --page raw \
        --csv \
        "${BUILD_DIR}/dl_benchmarks" "${args[@]}" \
        > "${csv}" 2>> "${log}"; then
        echo "ncu_profile : ${name} metrics failed" >&2
        cat "${log}" >&2
        return 1
    fi

    print_ncu_summary "${csv}" "${name}"
}

print_ncu_summary() {
    local csv="$1"
    local name="$2"

    if ! command -v python3 >/dev/null 2>&1; then
        echo "ncu_summary : ${name}"
        echo "python3 not found; raw CSV saved to ${csv}"
        return
    fi

    python3 - "${csv}" "${name}" <<'PY'
import csv
import sys

path = sys.argv[1]
benchmark = sys.argv[2]

def compact_kernel(name):
    if not name:
        return "<unknown>"
    if "::" in name:
        name = name.split("::")[-1]
    name = name.replace("(const float *, float *, unsigned long, unsigned long)", "")
    name = name.replace("(float *, const float *, int, int)", "")
    name = name.replace("(const float *, const float *, float *, int)", "")
    name = name.replace("(const float *, float *, unsigned long)", "")
    name = name.replace("(const float *, const float *, float *, unsigned long)", "")
    name = name.replace("void ", "")
    return name.strip()

def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def pick(row, names):
    for name in names:
        if name in row and row[name] != "":
            return row[name]
    return ""

with open(path, newline="") as f:
    rows = list(csv.reader(f))

header_index = None
for i, row in enumerate(rows):
    if "Kernel Name" in row or "launch__kernel_name" in row:
        header_index = i
        break

if header_index is None or len(rows) <= header_index + 2:
    print(f"ncu_summary : {benchmark}")
    print(f"  no metric rows found in {path}")
    sys.exit(0)

header = rows[header_index]
data_rows = rows[header_index + 2:]  # next row is the units row

print(f"ncu_summary : {benchmark}")
print(f"{'kernel':64} {'time_us':>12} {'dram_%':>10} {'sm_%':>10} {'l2_%':>10} {'l1_%':>10}")
print("-" * 122)

for values in data_rows:
    if not values or all(v == "" for v in values):
        continue
    if values[0].startswith("=="):
        break
    row = {header[i]: values[i] if i < len(values) else "" for i in range(len(header))}

    kernel = compact_kernel(pick(row, ["launch__kernel_name", "Kernel Name"]))
    runtime_ns = to_float(pick(row, ["gpu__time_duration.sum", "gpu__time_duration.avg"]))
    runtime_us = runtime_ns / 1000.0 if runtime_ns is not None else None
    dram = to_float(pick(row, ["dram__throughput.avg.pct_of_peak_sustained_elapsed"]))
    sm = to_float(pick(row, ["sm__throughput.avg.pct_of_peak_sustained_elapsed"]))
    l2 = to_float(pick(row, ["lts__throughput.avg.pct_of_peak_sustained_elapsed"]))
    l1 = to_float(pick(row, ["l1tex__throughput.avg.pct_of_peak_sustained_elapsed"]))

    def fmt(value):
        return f"{value:10.2f}" if value is not None else f"{'n/a':>10}"

    print(f"{kernel[:64]:64} {fmt(runtime_us)} {fmt(dram)} {fmt(sm)} {fmt(l2)} {fmt(l1)}")
PY
}

echo "configure : ${BUILD_DIR}"
cmake -S . -B "${BUILD_DIR}" -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}"

echo
echo "build : tensor_test dl_benchmarks"
cmake --build "${BUILD_DIR}" --target tensor_test dl_benchmarks

echo
echo "tests : start"
ctest --test-dir "${BUILD_DIR}" --output-on-failure
echo "tests : passed"

if command -v ncu >/dev/null 2>&1; then
    clean_ld_library_path
    prefer_wsl_cuda_driver
    mkdir -p "${REPORT_DIR}"
    echo
    echo "ncu : start"
    print_cuda_driver_diagnostics
    for benchmark in ${BENCHMARKS}; do
        if ! profile_benchmark "${benchmark}"; then
            echo "ncu_profile : ${benchmark} failed"
            echo "hint : check that LD_LIBRARY_PATH does not contain a CUDA stubs directory"
            echo "hint : run 'echo \$LD_LIBRARY_PATH' and remove paths like /usr/local/cuda/lib64/stubs"
            echo "hint : run 'ldconfig -p | grep libcuda' and verify it points to the real NVIDIA driver library"
            exit 1
        fi
    done
    echo
    echo "ncu : passed"
else
    echo
    echo "ncu : not found, skipping Nsight Compute reports"
fi
