import requests

ride = {
    "text": "I am a boy"
}

url = 'http://127.0.0.1:9696/classify'
response = requests.post(url, json=ride)
print(response.json())
