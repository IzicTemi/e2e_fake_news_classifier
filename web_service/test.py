import requests

text = """
            In fact, every generation fears the death of literacy at the hands of some new media technology.
            And yet I’m here to share some optimism.
            After long existence as a confirmed cynic who shared the general belief in our imminent cultural doom,
            I felt an unfamiliar sensation 15 years ago when the Internet came over the horizon: I found myself becoming excited and hopeful.
        """
event = {
    'text': text,
}

url = ''
response = requests.post(url, json=event, timeout=300).json()
print(response)
