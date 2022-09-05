import os

# import sys
from pathlib import Path

from prefect import flow
from prefect.task_runners import SequentialTaskRunner

import get_data

# lambda_function_path = str(Path.cwd().parent.absolute().joinpath('web_service'))

# sys.path.insert(0, lambda_function_path)

# from lambda_function import classify, lambda_handler

# classify("ball")


@flow(task_runner=SequentialTaskRunner)
def test_get_data():
    DATA_PATH = get_data.DATA_PATH
    Path(DATA_PATH).mkdir(parents=True, exist_ok=True)
    dataset = 'tarundalal/100-richest-people-in-world'
    data_ft = get_data.download_dataset.submit(dataset)
    assert '100-richest-people-in-world.zip' in os.listdir(DATA_PATH)
    unzip_ft = get_data.unzip.submit(wait_for=[data_ft])
    assert 'TopRichestInWorld.csv' in os.listdir(DATA_PATH)
    get_data.del_zip.submit(wait_for=[unzip_ft])
    assert '100-richest-people-in-world.zip' not in os.listdir(DATA_PATH)
    os.remove(f"{DATA_PATH}/TopRichestInWorld.csv")


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
