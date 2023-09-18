from model import LLaMA2


if __name__ == "__main__":
    device = "auto"
    model_size = "7b"
    model = LLaMA2(device=device, model_size=model_size)
    