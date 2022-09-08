# pylint: disable=import-error
# pylint: disable=no-name-in-module

import os
import shutil
from pathlib import Path

import numpy as np
from prefect import flow
from prefect.task_runners import SequentialTaskRunner
from tensorflow.keras.preprocessing import sequence

import get_data
from web_service import prepare
from web_service.model_server import ModelServer


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


def test_prepare():
    MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI')
    model_name = os.getenv('MODEL_NAME')
    prepare.main(MLFLOW_TRACKING_URI, model_name)

    assert Path('./artifact').exists()
    shutil.rmtree('./artifact')


class MockModel:
    def __init__(self):
        self.text = None

    def predict(self, text):
        self.text = text
        return np.array([[0]])


class MockTokenizer:
    def __init__(self):
        self.list_of_text = None

    def texts_to_sequences(self, list_of_text):
        self.list_of_text = list_of_text
        return np.array([[1, 2, 3]])


def test_prepare_tokens():
    model = MockModel()
    tokenizer = MockTokenizer()

    model_service = ModelServer(model, tokenizer)
    test_text = "The big brown dog"
    prepped_tokens = model_service.prepare_tokens(test_text)

    assert (
        prepped_tokens.all()
        == sequence.pad_sequences(np.array([[1, 2, 3]]), maxlen=300).all()
    )


def test_classify():
    model = MockModel()
    tokenizer = MockTokenizer()

    model_service = ModelServer(model, tokenizer)

    test_prepped_tokens = np.array(
        [
            0,
            0,
            0,
            0,
            817,
            308,
            69,
            5128,
            1,
            2176,
            2814,
            84,
            20,
            68,
            160,
            479,
            152,
            817,
            308,
            1243,
            176,
            15,
            1,
            23,
            14,
            4359,
            2305,
            9439,
            6,
            3121,
            7556,
            351,
            160,
            2814,
            308,
            1351,
            259,
            287,
            10,
            360,
            603,
            3490,
            3660,
            125,
            308,
            284,
            302,
            1,
            622,
            738,
            68,
            160,
            2132,
            947,
            978,
            5831,
            308,
            12,
            1050,
            589,
            1,
            9038,
            537,
            2057,
            1041,
            455,
            590,
            1619,
            200,
            955,
            5156,
            494,
            2464,
            3527,
            75,
            3648,
            532,
            724,
            252,
            2069,
            5110,
            1405,
            18,
            1095,
            230,
            355,
            1405,
            18,
            9968,
            538,
            355,
            738,
            308,
            1016,
            6302,
            5,
            1,
            22,
            784,
            77,
            966,
            15,
            2585,
            5,
            7238,
            1122,
            6,
            953,
            39,
            15,
            83,
            6,
            953,
            87,
            2087,
            1420,
            280,
            4359,
            2882,
            308,
            54,
            3091,
            242,
            210,
            1528,
            1,
            648,
            6,
            1207,
            145,
            4202,
            77,
            3660,
            3304,
            106,
            6,
            107,
            64,
            1310,
            1,
            170,
            727,
            308,
            1006,
            1883,
            266,
            2,
            1,
            308,
            107,
            1528,
            155,
            75,
            20,
            365,
            1223,
            167,
            391,
            308,
            1016,
            856,
            1256,
            5640,
            308,
            9081,
            1,
            4961,
            1643,
            7489,
            185,
            398,
            745,
            372,
            457,
            41,
            107,
            3315,
            20,
            308,
            5527,
            392,
            83,
            39,
            275,
            101,
            286,
            5,
            23,
            14,
            77,
            6,
            181,
            2936,
            946,
            110,
            6062,
            470,
            146,
            210,
            970,
            121,
            111,
            118,
            210,
            3660,
            12,
            125,
            398,
            160,
            236,
            2333,
            978,
            1681,
            2333,
            5124,
            308,
            1023,
            2,
            167,
            96,
            63,
            75,
            372,
            279,
            5593,
            862,
            365,
            3346,
            4139,
            308,
            2333,
            1586,
            2652,
            1374,
            20,
            53,
            142,
            2246,
            3000,
            64,
            1,
            2019,
            727,
            176,
            351,
            160,
            170,
            3666,
            75,
            160,
            2019,
            20,
            33,
            727,
            48,
            6119,
            351,
            4317,
            143,
            3185,
            569,
            6,
            625,
            75,
            160,
            63,
            1670,
            77,
            8111,
            1780,
            2936,
            79,
            160,
            657,
            106,
            181,
            392,
            320,
            4838,
            5693,
            1183,
            2563,
            1196,
            106,
            456,
            77,
            6159,
            1,
            625,
            146,
            4815,
            4384,
            1361,
            486,
            582,
            351,
            156,
            112,
            78,
            3215,
        ],
        dtype='int32',
    )

    actual_prediction = model_service.classify(test_prepped_tokens)
    expected_prediction = np.array([[0]])

    assert actual_prediction == expected_prediction


def test_lambda_handler():
    model = MockModel()
    tokenizer = MockTokenizer()

    model_service = ModelServer(model, tokenizer)

    event = {
        "text": "The big brown fox",
    }

    actual_predictions = model_service.lambda_handler(event)

    expected_predictions = {'text': event['text'], 'class': False}

    assert actual_predictions == expected_predictions
