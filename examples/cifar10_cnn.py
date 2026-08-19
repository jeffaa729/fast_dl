import dl


def main() -> None:
    device = dl.cuda(0)
    model = dl.Sequential()
    model.add_conv2d(3, 16, 3, device, padding=1)
    model.add_relu()
    model.add_max_pool2d(2)
    model.add_conv2d(16, 32, 3, device, padding=1)
    model.add_relu()
    model.add_max_pool2d(2)

    x = dl.Tensor.randn((8, 3, 32, 32), dl.float32, device)
    y = model(x)

    print("cifar10_cnn_py_demo : start")
    print(f"parameters : {model.num_parameters()}")
    print(f"features : shape={y.shape.dims}")
    print("cifar10_cnn_py_demo : passed")


if __name__ == "__main__":
    main()
