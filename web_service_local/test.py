import requests

ride = {"text": "I am a boy"}

url = 'http://127.0.0.1:9696/classify'
response = requests.post(url, json=ride, timeout=300)
print(response.json())
