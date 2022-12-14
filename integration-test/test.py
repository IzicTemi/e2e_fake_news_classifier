import requests
from deepdiff import DeepDiff

text = "I am a boy"
input_dict = {
    'text': text,
}

url = 'http://localhost:8080/2015-03-31/functions/function/invocations'
actual_response = requests.post(url, json=input_dict, timeout=300).json()
print(actual_response)

expected_response = {
    'text': text,
    'class': False,
}


diff = DeepDiff(actual_response, expected_response, significant_digits=1)
print(f'diff={diff}')

assert 'type_changes' not in diff
assert 'values_changed' not in diff
