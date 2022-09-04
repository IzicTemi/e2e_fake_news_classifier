# pylint: disable=import-error
# pylint: disable=no-name-in-module

import os
import json
import pickle
from pathlib import Path

import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
from tensorflow.keras.preprocessing import sequence

MLFLOW_TRACKING_URI = os.environ['MLFLOW_TRACKING_URI']
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)


def get_tokenizer():
    with open('./artifact/tokenizer.bin', "rb") as f_in:
        tokenizer = pickle.load(f_in)

    return tokenizer


def prepare(text):
    maxlen = 300
    tokenizer = get_tokenizer()
    tokens = tokenizer.texts_to_sequences(text)
    tokens = sequence.pad_sequences(tokens, maxlen=maxlen)
    return tokens


def load_model():
    uri_path = Path.cwd().joinpath('artifact/model').as_uri()
    print(uri_path)
    for file in os.scandir(Path.cwd().joinpath('artifact/model')):
        print(file)
    model = mlflow.keras.load_model(uri_path)
    return model


def classify(text):
    print("Loading the model...")
    model = load_model()
    print("Successful!")
    preds = model.predict(text)
    return preds


def lambda_handler(event, context):
    # pylint: disable=unused-argument
    # pylint: disable=unused-variable
    text = event['text']

    prepped_text = prepare(text)

    pred = classify(prepped_text)

    result = {
        'text': text,
        'class': 'boy',
    }

    return json.dumps(result)
