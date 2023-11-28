import requests
from PIL import Image
import pickle
import codecs
from typing import Any


url = "http://127.0.0.1:8085"
img_path = "data/HOMER.png"
promt = """Please give me an answer to every question"""


def obj2str(obj: Any):
    raw_data = codecs.encode(pickle.dumps(
        obj, protocol=pickle.HIGHEST_PROTOCOL), "base64").decode('latin1')
    return raw_data


def main():
    image = Image.open(img_path)
    raw_image = obj2str(image)
    data = {"prompt": promt, "image": raw_image}

    request = requests.post(url+"/generate", json=data).json()
    print(request)


if __name__ == "__main__":
    main()
