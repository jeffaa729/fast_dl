#!/usr/bin/env python3

import csv
import math
import os
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as error:
    print(f"pytorch_mnist_mlp_benchmark_error : {error}", file=sys.stderr)
    sys.exit(1)


@dataclass
class Metrics:
    loss: float = 0.0
    accuracy: float = 0.0


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def env_non_negative_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed >= 0 else default


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = float(value)
    except ValueError:
        return default
    return parsed if parsed > 0.0 else default


def env_string(name: str, default: str) -> str:
    return os.environ.get(name, default)


def read_images(path: str) -> torch.Tensor:
    with open(path, "rb") as file:
        magic, count, rows, cols = struct.unpack(">IIII", file.read(16))
        if magic != 2051:
            raise RuntimeError(f"MNIST image file has wrong magic number: {path}")
        data = file.read(count * rows * cols)
    tensor = torch.frombuffer(bytearray(data), dtype=torch.uint8)
    return tensor.reshape(count, rows * cols).to(torch.float32).div_(255.0)


def read_labels(path: str) -> torch.Tensor:
    with open(path, "rb") as file:
        magic, count = struct.unpack(">II", file.read(8))
        if magic != 2049:
            raise RuntimeError(f"MNIST label file has wrong magic number: {path}")
        data = file.read(count)
    return torch.frombuffer(bytearray(data), dtype=torch.uint8).to(torch.long)


class MnistBatches:
    def __init__(self, images: torch.Tensor, labels: torch.Tensor, batch_size: int):
        if images.shape[0] != labels.shape[0]:
            raise RuntimeError("MNIST image and label counts do not match")
        self.images = images
        self.labels = labels
        self.batch_size = batch_size

    def __len__(self) -> int:
        return self.images.shape[0]

    def batches(self, max_batches: int = 0):
        count = len(self)
        batch_index = 0
        for start in range(0, count, self.batch_size):
            if max_batches > 0 and batch_index >= max_batches:
                break
            end = min(start + self.batch_size, count)
            yield self.images[start:end], self.labels[start:end]
            batch_index += 1


class MnistMLP(nn.Module):
    def __init__(self, init: str):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
        self.apply_init(init)

    def apply_init(self, init: str) -> None:
        for module in (self.fc1, self.fc2):
            if init == "xavier":
                nn.init.xavier_uniform_(module.weight)
            else:
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
            nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


def evaluate(model: MnistMLP,
             loader: MnistBatches,
             device: torch.device,
             max_batches: int) -> Metrics:
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images_cpu, labels_cpu in loader.batches(max_batches):
            images = images_cpu.to(device)
            labels = labels_cpu.to(device)
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            batch_size = labels.shape[0]
            loss_sum += float(loss.detach().cpu()) * batch_size
            correct += int((logits.argmax(dim=1) == labels).sum().detach().cpu())
            total += batch_size

    model.train()
    if total == 0:
        return Metrics()
    return Metrics(loss_sum / total, correct / total)


def train_one_epoch(model: MnistMLP,
                    loader: MnistBatches,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    max_batches: int) -> int:
    model.train()
    batches = 0
    for images_cpu, labels_cpu in loader.batches(max_batches):
        images = images_cpu.to(device)
        labels = labels_cpu.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        batches += 1
    return batches


def load_data(batch_size: int) -> Tuple[MnistBatches, MnistBatches]:
    train = MnistBatches(
        read_images("data/mnist/train-images-idx3-ubyte"),
        read_labels("data/mnist/train-labels-idx1-ubyte"),
        batch_size,
    )
    test = MnistBatches(
        read_images("data/mnist/t10k-images-idx3-ubyte"),
        read_labels("data/mnist/t10k-labels-idx1-ubyte"),
        batch_size,
    )
    return train, test


def write_csv(path: str,
              total_ms: float,
              epochs: int,
              batch_size: int,
              train_batches: int) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    avg_epoch_ms = total_ms / float(epochs)
    with output_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "name",
            "epochs",
            "batch_size",
            "train_batches",
            "total_ms",
            "avg_epoch_ms",
        ])
        writer.writerow([
            "pytorch_mnist_mlp",
            epochs,
            batch_size,
            train_batches,
            total_ms,
            avg_epoch_ms,
        ])


def main() -> int:
    if not torch.cuda.is_available():
        print("pytorch_mnist_mlp_benchmark_error : CUDA is not available", file=sys.stderr)
        return 1

    epochs = env_int("MNIST_EPOCHS", 10)
    batch_size = env_int("MNIST_BATCH_SIZE", 64)
    max_train_batches = env_non_negative_int("MNIST_MAX_TRAIN_BATCHES", 0)
    max_test_batches = env_non_negative_int("MNIST_MAX_TEST_BATCHES", 0)
    train_eval_batches = env_non_negative_int("MNIST_TRAIN_EVAL_BATCHES", 20)
    learning_rate = env_float("MNIST_LR", 0.01)
    init = env_string("MNIST_LINEAR_INIT", "kaiming")
    csv_path = env_string(
        "PYTORCH_MNIST_MLP_CSV",
        "benchmark_results/pytorch_mnist_mlp_training.csv",
    )

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.manual_seed(1234)

    train_loader, test_loader = load_data(batch_size)
    model = MnistMLP(init).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    print("pytorch_mnist_mlp_benchmark : start")
    print(
        f"config : epochs={epochs} batch_size={batch_size} lr={learning_rate} "
        f"init={init} train_eval_batches={train_eval_batches} "
        f"train_samples={len(train_loader)} test_samples={len(test_loader)}"
    )

    torch.cuda.synchronize()
    start = time.perf_counter()
    trained_batches = 0
    last_train = Metrics()
    last_test = Metrics()
    for epoch in range(1, epochs + 1):
        trained_batches = train_one_epoch(
            model, train_loader, optimizer, device, max_train_batches)
        last_train = evaluate(model, train_loader, device, train_eval_batches)
        last_test = evaluate(model, test_loader, device, max_test_batches)
        print(
            f"epoch {epoch} : batches={trained_batches} "
            f"train_loss={last_train.loss:.4f} train_acc={last_train.accuracy:.4f} "
            f"test_loss={last_test.loss:.4f} test_acc={last_test.accuracy:.4f}"
        )

    torch.cuda.synchronize()
    total_ms = (time.perf_counter() - start) * 1000.0

    if not (math.isfinite(last_train.loss) and math.isfinite(last_test.loss)):
        raise RuntimeError("training produced non-finite loss")

    write_csv(csv_path, total_ms, epochs, batch_size, trained_batches)
    print(f"total_ms : {total_ms:.4f}")
    print(f"avg_epoch_ms : {total_ms / float(epochs):.4f}")
    print(f"pytorch_mnist_mlp_csv : {csv_path}")
    print("pytorch_mnist_mlp_benchmark : passed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"pytorch_mnist_mlp_benchmark_error : {error}", file=sys.stderr)
        print("pytorch_mnist_mlp_benchmark : not passed")
        raise SystemExit(1)
