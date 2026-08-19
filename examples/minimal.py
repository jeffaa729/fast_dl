import dl


def main() -> None:
    device = dl.cuda(0)
    x = dl.Tensor.randn((2, 3), dl.float32, device)
    y = dl.relu(x)

    print("minimal_demo : start")
    print(f"x : shape={x.shape.dims} dtype={x.dtype} device={x.device}")
    print(f"y : shape={y.shape.dims}")
    print("minimal_demo : passed")


if __name__ == "__main__":
    main()
