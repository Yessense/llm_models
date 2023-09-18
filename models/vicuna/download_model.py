from model import Vicuna


if __name__ == "__main__":
    device = "auto"
    model_size = "7b"
    model = Vicuna(device=device, model_size=model_size)
    