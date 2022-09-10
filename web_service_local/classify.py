# pylint: disable=import-error
# pylint: disable=no-name-in-module

import os
import pickle
from pathlib import Path

import mlflow
import requests
import mlflow.pyfunc
from flask import Flask, jsonify, request
from pymongo import MongoClient
from mlflow.tracking import MlflowClient
from tensorflow.keras.preprocessing import sequence

MLFLOW_TRACKING_URI = os.environ['MLFLOW_TRACKING_URI']
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

EVIDENTLY_SERVICE_ADDRESS = os.getenv('EVIDENTLY_SERVICE')
MONGODB_ADDRESS = os.getenv("MONGODB_ADDRESS")


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
mongo_client = MongoClient(MONGODB_ADDRESS)
db = mongo_client.get_database("prediction_service")
collection = db.get_collection("data")


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

    if MONGODB_ADDRESS != '':

        save_to_db(text, int(final_pred is True))

    return jsonify(result)


def save_to_db(record, prediction):
    rec = record.copy()
    rec['prediction'] = prediction
    collection.insert_one(rec)


def send_to_evidently_service(record, prediction):
    rec = record.copy()
    rec['prediction'] = prediction
    requests.post(f"{EVIDENTLY_SERVICE_ADDRESS}/iterate/faker", json=[rec], timeout=300)


if __name__ == "__main__":

    app.run(debug=True, host='0.0.0.0', port=9696)
