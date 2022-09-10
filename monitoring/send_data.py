import os
import json
import uuid
from time import sleep

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


path = os.getenv('DATA_PATH')

rel_path = f"../{path}"

data = read_data(rel_path)

url = 'http://127.0.0.1:9696/classify'

with open("target.csv", 'a', encoding='utf-8') as f_target:
    for index, row in data.iterrows():
        row['id'] = str(uuid.uuid4())
        f_target.write(f"{row['id']},{row['category']}\n")

        event = {'text': row['text']}
        resp = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            data=json.dumps(event),
            timeout=600,
        ).json()
        print(f"class: {int(resp['class'] is True)}")
        sleep(1)
