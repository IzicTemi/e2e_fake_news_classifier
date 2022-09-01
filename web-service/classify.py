import os
import pickle

import mlflow
from mlflow.tracking import MlflowClient

from flask import Flask, request, jsonify

import mlflow.pyfunc
from tensorflow.keras.preprocessing import text, sequence

from pathlib import Path

MLFLOW_TRACKING_URI=os.environ['MLFLOW_TRACKING_URI']
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
    
def load_model():
    uri_path = Path.cwd().joinpath('artifact/model').as_uri()
    model = mlflow.keras.load_model(uri_path)
    return model
    

def classify(text):
    model = load_model()
    preds = model.predict(text)
    return preds


app = Flask('fake-news-classifier')


@app.route('/classify', methods=['POST'])
def predict_endpoint():
    text = request.get_json()

    prepped_text = prepare(text['text'])
    pred = classify(prepped_text)

    result = {
        'text': text['text'],
        'class': pred
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
