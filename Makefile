BUILD_DIR ?= build
CMAKE ?= cmake
CTEST ?= ctest
BUILD_TYPE ?= Release
PYTHON ?= python3

.PHONY: all configure build test tests benchmarks ops_benchmark dl_benchmarks python-install python-test python-examples python-benchmark demos mnist cifar clean clean-generated

all: build

build: configure
	$(CMAKE) --build $(BUILD_DIR)

configure:
	$(CMAKE) -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(BUILD_TYPE)

test: tests

tests: build
	$(CTEST) --test-dir $(BUILD_DIR) --output-on-failure

benchmarks: configure
	$(CMAKE) --build $(BUILD_DIR) --target ops_benchmark dl_benchmarks

ops_benchmark: configure
	$(CMAKE) --build $(BUILD_DIR) --target ops_benchmark
	./$(BUILD_DIR)/ops_benchmark

dl_benchmarks: configure
	$(CMAKE) --build $(BUILD_DIR) --target dl_benchmarks
	./$(BUILD_DIR)/dl_benchmarks

python-install:
	$(PYTHON) -m pip install -e ".[test,benchmark]"

python-test: python-install
	$(PYTHON) -m pytest tests/python

python-examples: python-install
	$(PYTHON) examples/minimal.py
	$(PYTHON) examples/mnist_mlp.py
	$(PYTHON) examples/cifar10_cnn.py
	$(PYTHON) examples/tiny_gpt.py

python-benchmark: benchmarks python-install
	$(PYTHON) benchmarks/python/dl_ops_benchmark.py
	$(PYTHON) benchmarks/python/pytorch_ops_benchmark.py
	$(PYTHON) benchmarks/python/compare_ops.py

demos: configure
	$(CMAKE) --build $(BUILD_DIR) --target mnist_mlp_demo cifar10_cnn_demo

mnist: configure
	$(CMAKE) --build $(BUILD_DIR) --target mnist_mlp_demo
	./$(BUILD_DIR)/mnist_mlp_demo

cifar: configure
	$(CMAKE) --build $(BUILD_DIR) --target cifar10_cnn_demo
	./$(BUILD_DIR)/cifar10_cnn_demo

clean:
	$(CMAKE) -E rm -rf $(BUILD_DIR)

clean-generated:
	$(CMAKE) -E rm -rf build build-* build-nmake
	$(CMAKE) -E rm -rf benchmark_results ncu_reports outputs
	$(CMAKE) -E rm -rf python-env python-envVENV_DIR=python-env .venv
	$(CMAKE) -E rm -rf .cuda-driver-lib
	$(CMAKE) -E rm -f *.nsys-rep *.ncu-rep *.sqlite
