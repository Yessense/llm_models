import requests
import base64
from PIL import Image
# import numpy as np
import pickle
import codecs
from typing import Any 

url = "http://127.0.0.1:8083"
img_path = "data/2_200.png"
promt = """Describe image in detail"""


def obj2str(obj: Any):
    raw_data = codecs.encode(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL), "base64").decode('latin1')
    return raw_data

def main():
    image = Image.open(img_path)
    raw_image = obj2str(image)
    data = {"goal": promt, "image": raw_image, "task_type": 1}
    
    # image = Image.open(img_path)
    # arr_image = np.array(image).tolist()
    # data = {"text": promt, "image": None}
    
    request = requests.post(url+"/add_llp_task", json=data).json()
    # model_name = requests.get(url+"/name").json()
    
    print(request)
    # print(model_name)

if __name__ == "__main__":
    main()
