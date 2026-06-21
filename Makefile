# Thin wrapper around the CMake build.

BUILD_DIR ?= build
CMAKE ?= cmake

.PHONY: all configure test clean

all: configure
	$(CMAKE) --build $(BUILD_DIR)

configure:
	$(CMAKE) -S . -B $(BUILD_DIR)

test: all
	$(CMAKE) --build $(BUILD_DIR) --target test_layers test_oop test_network test_train_mnist
	$(CMAKE) --build $(BUILD_DIR) --target test

clean:
	$(CMAKE) -E rm -rf $(BUILD_DIR)
