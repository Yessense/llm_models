import requests
url = "http://127.0.0.1:8080/generate"
data = {'prompt': 'pick up fruit from the table'}
print(requests.post(url, json=data).json()['output'])


