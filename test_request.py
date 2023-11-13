import requests
import base64
from PIL import Image

url = "http://127.0.0.1:8080"
img_path = "data/1-sobaki-1.jpg"
promt = """What does it mean to love an animanl?"""

def main():
    with open(img_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    
    print("With decode")
    data = {"text": promt, "image": encoded_string}
    
    request = requests.post(url+"/generate", json=data).json()
    model_name = requests.get(url+"/name").json()
    
    print(request)
    print(model_name)

if __name__ == "__main__":
    main()
