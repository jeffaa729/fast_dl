import math
from dataclasses import dataclass
from pathlib import Path

import dl
import numpy as np


EPOCHS = 1
BATCH_SIZE = 64
LEARNING_RATE = 0.01
SAMPLE_COUNT = 10
DATA_DIR = Path("data/cifar-10-batches-bin")
CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


@dataclass
class Batch:
    images: dl.Tensor
    labels: dl.Tensor
    size: int


class Cifar10Dataset:
    def __init__(self, root: Path, train: bool) -> None:
        self.images = []
        self.labels = []
        paths = [root / f"data_batch_{i}.bin" for i in range(1, 6)] if train else [root / "test_batch.bin"]
        for path in paths:
            self._read_batch_file(path)
        self.images = np.ascontiguousarray(np.concatenate(self.images, axis=0))
        self.labels = np.ascontiguousarray(np.concatenate(self.labels, axis=0))

    def __len__(self) -> int:
        return len(self.labels)

    def batch(self, start: int, batch_size: int, device: dl.Device) -> Batch:
        end = min(start + batch_size, len(self))
        images = np.ascontiguousarray(self.images[start:end].astype(np.float32) / 255.0)
        labels = np.ascontiguousarray(self.labels[start:end])

        return Batch(
            dl.Tensor.from_numpy(images, device),
            dl.Tensor.from_numpy(labels, device),
            end - start,
        )

    def _read_batch_file(self, path: Path) -> None:
        record_size = 1 + 3 * 32 * 32
        raw = path.read_bytes()
        if len(raw) % record_size != 0:
            raise RuntimeError(f"bad CIFAR-10 file size: {path}")

        records = np.frombuffer(raw, dtype=np.uint8).reshape(-1, record_size)
        self.labels.append(records[:, 0].astype(np.int64))
        self.images.append(records[:, 1:].reshape(-1, 3, 32, 32))


class Cifar10DataLoader:
    def __init__(self, root: Path, train: bool, batch_size: int, device: dl.Device) -> None:
        self.dataset = Cifar10Dataset(root, train)
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


class Cifar10Cnn:
    def __init__(self, device: dl.Device) -> None:
        self.conv1 = dl.Conv2D(3, 16, 3, device, stride=1, padding=1)
        self.relu1 = dl.ReLU()
        self.pool1 = dl.MaxPool2D(2)
        self.conv2 = dl.Conv2D(16, 32, 3, device, stride=1, padding=1)
        self.relu2 = dl.ReLU()
        self.pool2 = dl.MaxPool2D(2)
        self.fc = dl.Linear(32 * 8 * 8, 10, device)

    def __call__(self, x: dl.Tensor) -> dl.Tensor:
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = dl.flatten(x)
        return self.fc(x)

    def parameters(self):
        return self.conv1.parameters() + self.conv2.parameters() + self.fc.parameters()

    def num_parameters(self) -> int:
        return self.conv1.num_parameters() + self.conv2.num_parameters() + self.fc.num_parameters()


def class_name(label: int) -> str:
    return CLASS_NAMES[label] if 0 <= label < len(CLASS_NAMES) else "unknown"


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


def evaluate(model: Cifar10Cnn, loader: Cifar10DataLoader):
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


def train_one_epoch(model: Cifar10Cnn, loader: Cifar10DataLoader, optimizer: dl.SGD) -> int:
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


def print_sample_results(model: Cifar10Cnn, loader: Cifar10DataLoader, sample_count: int) -> None:
    loader.reset()
    batch = loader.next()
    with dl.no_grad():
        predictions = predict_classes(model(batch.images))
    labels = batch.labels.to_int64_list()

    for index in range(min(sample_count, len(labels))):
        print(
            f"sample {index} : predicted={class_name(predictions[index])} "
            f"actual={class_name(labels[index])}"
        )


def main() -> None:
    passed = True
    try:
        device = dl.cuda(0)
        train_loader = Cifar10DataLoader(DATA_DIR, True, BATCH_SIZE, device)
        test_loader = Cifar10DataLoader(DATA_DIR, False, BATCH_SIZE, device)
        model = Cifar10Cnn(device)
        optimizer = dl.SGD(model.parameters(), LEARNING_RATE)

        print("cifar10_cnn_py_demo : start")
        print(f"parameters : {model.num_parameters()}")
        print(
            f"config : epochs={EPOCHS} batch_size={BATCH_SIZE} "
            f"lr={LEARNING_RATE} conv_init=kaiming linear_init=kaiming "
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
        print(f"cifar10_cnn_py_demo_error : {error}")
        passed = False

    print(f"cifar10_cnn_py_demo : {'passed' if passed else 'not passed'}")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
