# pylint: disable=import-error
# pylint: disable=no-name-in-module

import os
import pickle
from pathlib import Path

import mlflow
import mlflow.pyfunc
from flask import Flask, jsonify, request
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
    tokens = tokenizer.texts_to_sequences([text])
    prepped_tokens = sequence.pad_sequences(tokens, maxlen=maxlen)
    return prepped_tokens


def load_model():
    uri_path = Path.cwd().joinpath('artifact/model').as_uri()
    model = mlflow.keras.load_model(uri_path)
    return model


def classify(prepped_tokens):
    print("Loading the model...")
    model = load_model()
    print("Successful!")
    preds = model.predict(prepped_tokens)
    return preds


app = Flask('fake-news-classifier')


@app.route('/classify', methods=['POST'])
def classify_endpoint():
    text = request.get_json()

    tokens = prepare(text['text'])

    pred = classify(tokens)

    int_pred = (pred > 0.5).astype("int32").tolist()[0][0]

    dict_map = {
        0: False,
        1: True,
    }

    final_pred = dict_map[int_pred]

    result = {
        'text': text['text'],
        'class': final_pred,
    }

    return jsonify(result)


if __name__ == "__main__":

    app.run(debug=True, host='0.0.0.0', port=9696)
