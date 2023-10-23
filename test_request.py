import requests
url = "http://127.0.0.1:8080/generate"
promt = """
    How would you hide a red apple on the table?
    1. move_to("red apple"), 2. pick_up("red apple", "unspecified"), 3. move_to("table"), 4. put("red apple", "table"), 5. done().
    How would you throw a red apple from the box on the chair?
    1. move_to("red apple"), 2. pick_up("red apple", "box"), 3. move_to("chair"), 4. put("red apple", "chair"), 5. done().
    How would you move a red apple from the table to the drawer?
        
"""


if __name__ == "__main__":
    data = {'prompt': promt}
    print(requests.post(url, json=data).json()['output'])
