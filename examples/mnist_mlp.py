import dl


class MnistMlp:
    def __init__(self, device: dl.Device) -> None:
        self.fc1 = dl.Linear(784, 128, device)
        self.relu = dl.ReLU()
        self.fc2 = dl.Linear(128, 10, device)

    def __call__(self, x: dl.Tensor) -> dl.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)

    def num_parameters(self) -> int:
        return self.fc1.num_parameters() + self.fc2.num_parameters()


def main() -> None:
    device = dl.cuda(0)
    model = MnistMlp(device)
    x = dl.Tensor.randn((64, 784), dl.float32, device)
    logits = model(x)

    print("mnist_mlp_py_demo : start")
    print(f"parameters : {model.num_parameters()}")
    print(f"logits : shape={logits.shape.dims}")
    print("mnist_mlp_py_demo : passed")


if __name__ == "__main__":
    main()
