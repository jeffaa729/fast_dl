BUILD_DIR ?= build
CMAKE ?= cmake
CTEST ?= ctest
BUILD_TYPE ?= Release

.PHONY: all configure build test tests benchmarks ops_benchmark dl_benchmarks demos mnist cifar clean clean-generated

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
