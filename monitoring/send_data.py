import os
import json
from time import sleep
from hashlib import sha1

import pandas as pd
import requests


def read_data(path):
    print("Loading data... ")
    true = pd.read_csv(f"{path}/True.csv")
    false = pd.read_csv(f"{path}/Fake.csv")

    true['category'] = 1
    false['category'] = 0

    df = pd.concat([true, false])

    df['text'] = df['title'] + "\n" + df['text']
    del df['title']
    del df['subject']
    del df['date']
    print("Successful!")

    df = df.sample(frac=1)

    return df


def compute_hash(text):
    return sha1(text.lower().encode('utf-8')).hexdigest()


path = os.getenv('DATA_PATH')

# rel_path = f"../{path}"

data = read_data(path)

url = 'http://127.0.0.1:9696/classify'

with open("target.csv", 'a', encoding='utf-8') as f_target:
    for index, row in data.iterrows():
        row['_id'] = compute_hash(row['text'])

        event = {'text': row['text']}
        resp = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            data=json.dumps(event),
            timeout=600,
        ).json()

        pred = int(resp['class'] is True)
        f_target.write(f"{row['_id']},{row['category']},\n")
        print(f"class: {pred}")
        sleep(0.5)
