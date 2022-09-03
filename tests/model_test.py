import os

# import sys
from pathlib import Path

import get_***REMOVED***

# lambda_function_path = str(Path.cwd().parent.absolute().joinpath('web_service'))

# sys.path.insert(0, lambda_function_path)

# from lambda_function import classify, lambda_handler

# classify("ball")


def test_get_***REMOVED***():
    DATA_PATH = get_***REMOVED***.DATA_PATH
    Path(DATA_PATH).mkdir(parents=True, exist_ok=True)
    ***REMOVED***set = 'tarundalal/100-richest-people-in-world'
    get_***REMOVED***.download_***REMOVED***set(***REMOVED***set)
    assert '100-richest-people-in-world.zip' in os.listdir(DATA_PATH)
    get_***REMOVED***.unzip()
    assert 'TopRichestInWorld.csv' in os.listdir(DATA_PATH)
    get_***REMOVED***.del_zip()
    assert '100-richest-people-in-world.zip' not in os.listdir(DATA_PATH)


# class ModelMock:
#     def __init__(self, value):
#         self.value = value

#     def classify(self, X):
#         n = len(X)
#         return [self.value] * n


# def test_predict():
#     model_mock = ModelMock(10.0)
#     model_service = model.ModelService(model_mock)

#     features = {
#         "PU_DO": "130_205",
#         "trip_distance": 3.66,
#     }

#     actual_prediction = model_service.predict(features)
#     expected_prediction = 10.0

#     assert actual_prediction == expected_prediction


# def test_lambda_handler():
#     model_mock = ModelMock(10.0)
#     model_version = 'Test123'
#     model_service = model.ModelService(model_mock, model_version)


#     event = {
#         "text": "I am a boy",
#         }

#     actual_predictions = model_service.lambda_handler(event)
#     expected_predictions = {
#         'text': event['text'],
#         'class': 'boy'
#     }

#     assert actual_predictions == expected_predictions
