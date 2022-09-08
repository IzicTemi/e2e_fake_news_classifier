import requests

text = "I am a boy"

input_dict = {
    'text': text,
}


url = 'http://127.0.0.1:9696/classify'
response = requests.post(url, json=input_dict, timeout=300)
print(response.json())
