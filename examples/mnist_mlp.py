import math
import struct
from dataclasses import dataclass
from pathlib import Path

import dl
import numpy as np


EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 0.01
SAMPLE_COUNT = 3
DATA_DIR = Path("data/mnist")


@dataclass
class Batch:
    images: dl.Tensor
    labels: dl.Tensor
    size: int


class MnistDataset:
    def __init__(self, image_path: Path, label_path: Path) -> None:
        self.images, self.rows, self.cols = self._read_images(image_path)
        self.labels = self._read_labels(label_path)
        if len(self.images) != len(self.labels):
            raise RuntimeError("MNIST image and label counts do not match")

    def __len__(self) -> int:
        return len(self.labels)

    def image_size(self) -> int:
        return self.rows * self.cols

    def batch(self, start: int, batch_size: int, device: dl.Device) -> Batch:
        end = min(start + batch_size, len(self))
        images = np.ascontiguousarray(self.images[start:end])
        labels = np.ascontiguousarray(self.labels[start:end])
        return Batch(
            dl.Tensor.from_numpy(images, device),
            dl.Tensor.from_numpy(labels, device),
            end - start,
        )

    @staticmethod
    def _read_images(path: Path):
        with path.open("rb") as f:
            magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
            if magic != 2051:
                raise RuntimeError(f"wrong MNIST image magic number: {magic}")
            raw = f.read(count * rows * cols)
            if len(raw) != count * rows * cols:
                raise RuntimeError(f"failed to read MNIST image data: {path}")

        images = np.frombuffer(raw, dtype=np.uint8)
        images = images.reshape(count, rows * cols).astype(np.float32)
        images /= 255.0
        return images, rows, cols

    @staticmethod
    def _read_labels(path: Path):
        with path.open("rb") as f:
            magic, count = struct.unpack(">II", f.read(8))
            if magic != 2049:
                raise RuntimeError(f"wrong MNIST label magic number: {magic}")
            raw = f.read(count)
            if len(raw) != count:
                raise RuntimeError(f"failed to read MNIST label data: {path}")
        return np.frombuffer(raw, dtype=np.uint8).astype(np.int64)


class MnistDataLoader:
    def __init__(self, image_path: Path, label_path: Path, batch_size: int, device: dl.Device) -> None:
        self.dataset = MnistDataset(image_path, label_path)
        self.batch_size = batch_size
        self.device = device
        self.cursor = 0

    def __len__(self) -> int:
        return len(self.dataset)

    def reset(self) -> None:
        self.cursor = 0

    def has_next(self) -> bool:
        return self.cursor < len(self.dataset)

    def next(self) -> Batch:
        batch = self.dataset.batch(self.cursor, self.batch_size, self.device)
        self.cursor += batch.size
        return batch


class MnistMlp:
    def __init__(self, device: dl.Device) -> None:
        self.fc1 = dl.Linear(784, 128, device)
        self.relu = dl.ReLU()
        self.fc2 = dl.Linear(128, 10, device)

    def __call__(self, x: dl.Tensor) -> dl.Tensor:
        return self.fc2(self.relu(self.fc1(x)))

    def parameters(self):
        return self.fc1.parameters() + self.fc2.parameters()

    def num_parameters(self) -> int:
        return self.fc1.num_parameters() + self.fc2.num_parameters()


def predict_classes(logits: dl.Tensor):
    values = logits.tolist()
    batch, classes = logits.shape.dims
    predictions = []
    for row in range(batch):
        offset = row * classes
        best = max(range(classes), key=lambda col: values[offset + col])
        predictions.append(best)
    return predictions


def count_correct(logits: dl.Tensor, labels: dl.Tensor) -> int:
    predictions = predict_classes(logits)
    targets = labels.to_int64_list()
    return sum(int(pred == target) for pred, target in zip(predictions, targets))


def evaluate(model: MnistMlp, loader: MnistDataLoader):
    loader.reset()
    loss_sum = 0.0
    correct = 0
    total = 0
    with dl.no_grad():
        while loader.has_next():
            batch = loader.next()
            logits = model(batch.images)
            loss = dl.cross_entropy(logits, batch.labels)
            loss_sum += loss.tolist()[0] * batch.size
            correct += count_correct(logits, batch.labels)
            total += batch.size
    return loss_sum / total, correct / total


def train_one_epoch(model: MnistMlp, loader: MnistDataLoader, optimizer: dl.SGD) -> int:
    loader.reset()
    batches = 0
    while loader.has_next():
        batch = loader.next()
        optimizer.zero_grad()
        logits = model(batch.images)
        loss = dl.cross_entropy(logits, batch.labels)
        loss.backward()
        optimizer.step()
        batches += 1
    return batches


def print_sample_results(model: MnistMlp, loader: MnistDataLoader, sample_count: int) -> None:
    loader.reset()
    batch = loader.next()
    with dl.no_grad():
        predictions = predict_classes(model(batch.images))
    labels = batch.labels.to_int64_list()

    text = "samples :"
    for index in range(min(sample_count, len(labels))):
        text += f" [{index}] pred={predictions[index]} label={labels[index]}"
    print(text)


def main() -> None:
    passed = True
    try:
        device = dl.cuda(0)
        train_loader = MnistDataLoader(
            DATA_DIR / "train-images-idx3-ubyte",
            DATA_DIR / "train-labels-idx1-ubyte",
            BATCH_SIZE,
            device,
        )
        test_loader = MnistDataLoader(
            DATA_DIR / "t10k-images-idx3-ubyte",
            DATA_DIR / "t10k-labels-idx1-ubyte",
            BATCH_SIZE,
            device,
        )
        model = MnistMlp(device)
        optimizer = dl.SGD(model.parameters(), LEARNING_RATE)

        print("mnist_mlp_py_demo : start")
        print(f"parameters : {model.num_parameters()}")
        print(
            f"config : epochs={EPOCHS} batch_size={BATCH_SIZE} "
            f"lr={LEARNING_RATE} init=kaiming "
            f"train_samples={len(train_loader)} test_samples={len(test_loader)}"
        )

        last_train_loss = math.nan
        last_train_acc = math.nan
        last_test_loss = math.nan
        last_test_acc = math.nan
        for epoch in range(1, EPOCHS + 1):
            batches = train_one_epoch(model, train_loader, optimizer)
            last_train_loss, last_train_acc = evaluate(model, train_loader)
            last_test_loss, last_test_acc = evaluate(model, test_loader)
            print(
                f"epoch {epoch} : batches={batches} "
                f"train_loss={last_train_loss:.4f} train_acc={last_train_acc:.4f} "
                f"test_loss={last_test_loss:.4f} test_acc={last_test_acc:.4f}"
            )

        print_sample_results(model, test_loader, SAMPLE_COUNT)
        passed = (
            math.isfinite(last_train_loss)
            and math.isfinite(last_test_loss)
            and 0.0 <= last_train_acc <= 1.0
            and 0.0 <= last_test_acc <= 1.0
        )
    except Exception as error:
        print(f"mnist_mlp_py_demo_error : {error}")
        passed = False

    print(f"mnist_mlp_py_demo : {'passed' if passed else 'not passed'}")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
