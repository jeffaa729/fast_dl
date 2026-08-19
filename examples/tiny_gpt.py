import dl


def main() -> None:
    device = dl.cuda(0)
    token_embeddings = dl.Tensor.randn((16, 64), dl.float32, device)
    projection = dl.Linear(64, 16, device)
    logits = projection(token_embeddings)

    print("tiny_gpt_py_demo : start")
    print(f"token_embeddings : shape={token_embeddings.shape.dims}")
    print(f"logits : shape={logits.shape.dims}")
    print("tiny_gpt_py_demo : placeholder passed")


if __name__ == "__main__":
    main()
