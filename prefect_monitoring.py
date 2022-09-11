# pylint: disable=import-error

import os
import json
import pickle
from pathlib import Path
from datetime import datetime

import mlflow
import pandas as pd
from prefect import flow, task, get_run_logger
from pymongo import MongoClient
from evidently import ColumnMapping
from mlflow.tracking import MlflowClient
from evidently.dashboard import Dashboard
from dateutil.relativedelta import relativedelta
from evidently.model_profile import Profile
from evidently.dashboard.tabs import (
    DataDriftTab,
    CatTargetDriftTab,
    ClassificationPerformanceTab,
)
from tensorflow.keras.preprocessing import sequence
from evidently.model_profile.sections import (
    DataDriftProfileSection,
    CatTargetDriftProfileSection,
    ClassificationPerformanceProfileSection,
)

from train import study_and_train
from get_data import get_data


@task
def upload_target(filename):
    logger = get_run_logger()
    logger.info("Simulating data in MongoDB")
    client = MongoClient("mongodb://localhost:27018/")
    collection = client.get_database("prediction_server").get_collection("data")
    with open(filename, encoding='utf-8') as f_target:
        for line in f_target.readlines():
            row = line.split(",")
            collection.update_one({"_id": row[0]}, {"$set": {"target": int(row[1])}})
    client.close()


@task
def prepare():
    logger = get_run_logger()
    logger.info("Getting Production Model artifacts")
    MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI')
    model_name = os.getenv('MODEL_NAME')
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    prod_version = client.get_latest_versions(name=model_name, stages=["Production"])[0]
    run_id = prod_version.run_id

    mlflow.artifacts.download_artifacts(run_id=run_id, dst_path='./artifact')


def get_tokenizer():
    with open('./artifact/tokenizer.bin', "rb") as f_in:
        tokenizer = pickle.load(f_in)

    return tokenizer


@task
def prepare_tokens(text):
    logger = get_run_logger()
    logger.info("Tokenizing... ")
    maxlen = 300
    tokenizer = get_tokenizer()
    tokens = tokenizer.texts_to_sequences(text)
    prepped_tokens = sequence.pad_sequences(tokens, maxlen=maxlen)
    return prepped_tokens


@task
def load_model():
    logger = get_run_logger()
    logger.info("Loading the model... ")
    uri_path = Path.cwd().joinpath('artifact/model').as_uri()
    model = mlflow.keras.load_model(uri_path)
    return model


@flow
def load_reference_data(path):
    logger = get_run_logger()
    logger.info("Loading the data... ")
    true = pd.read_csv(f"{path}/True.csv")
    false = pd.read_csv(f"{path}/Fake.csv")

    true['category'] = 1
    false['category'] = 0

    reference_data = pd.concat([true, false])

    reference_data['text'] = reference_data['title'] + "\n" + reference_data['text']
    reference_data['target'] = reference_data['category']

    reference_data = reference_data.sample(frac=0.05, random_state=1)

    init_prep = prepare.submit()
    model_future = load_model.submit(wait_for=[init_prep])
    model = model_future.result()
    prepped_tokens_future = prepare_tokens.submit(
        reference_data['text'], wait_for=[init_prep]
    )
    prepped_tokens = prepped_tokens_future.result()
    pred = model.predict(prepped_tokens)
    int_pred = (pred > 0.5).astype("int32")

    reference_data['prediction'] = int_pred

    del reference_data['category']
    del reference_data['title']
    del reference_data['subject']
    del reference_data['date']

    logger.info("Successful")

    return reference_data


@task
def fetch_data():
    logger = get_run_logger()
    logger.info("Fetching data from MongoDB")
    client = MongoClient("mongodb://localhost:27018/")
    data = client.get_database("prediction_server").get_collection("data").find()
    df = pd.DataFrame(list(data))
    return df


@task
def run_evidently(ref_data, data):
    logger = get_run_logger()
    logger.info("Running Evidently analysis... ")
    profile = Profile(
        sections=[
            DataDriftProfileSection(),
            ClassificationPerformanceProfileSection(),
            CatTargetDriftProfileSection(),
        ]
    )
    mapping = ColumnMapping(
        prediction="prediction",
        target="target",
        numerical_features=[],
        categorical_features=['text'],
        datetime_features=[],
    )
    profile.calculate(ref_data, data, mapping)

    dashboard = Dashboard(
        tabs=[
            DataDriftTab(),
            ClassificationPerformanceTab(verbose_level=0),
            CatTargetDriftTab(),
        ]
    )
    dashboard.calculate(ref_data, data, mapping)
    return json.loads(profile.json()), dashboard


@task
def save_report(result):
    logger = get_run_logger()
    logger.info("Saving report to MongoDB... ")
    client = MongoClient("mongodb://localhost:27018/")
    client.get_database("prediction_server").get_collection("report").insert_one(
        {'cat_target_drift': result[0]['cat_target_drift']}
    )
    client.get_database("prediction_server").get_collection("report").insert_one(
        {'classification_performance': result[0]['classification_performance']}
    )


@task
def save_html_report(result):
    logger = get_run_logger()
    logger.info("Saving report as HTML... ")
    result[1].save("evidently_report.html")


@flow(name='forced_training')
def start_train(date=None):
    logger = get_run_logger()
    if date is None:
        day = datetime.today()
        date = day.strftime("%Y-%m-%d")

    data_date = day - relativedelta(months=1)
    logger.info(f"Getting data for month: {str(data_date.month).zfill(2)}")
    get_data()
    logger.info(f"Training on: {date}")
    study_and_train()


@flow
def check_model_performance(result):
    '''
    This flow runs an analysis of the model using evidently and checks if the model's
    accuracy is worse by the reference accuracy by up to 10%.

    If yes, it triggers the training workflow.
    '''
    logger = get_run_logger()
    logger.info('Checking model Performance... ')
    # Get reference accuracy
    ref_accuracy = result[0]['classification_performance']['data']['metrics'][
        'reference'
    ]['accuracy']
    # Get current model accuracy
    curr_accuracy = result[0]['classification_performance']['data']['metrics'][
        'current'
    ]['accuracy']
    # Check if current model accuracy is worse than reference by up to 0.1
    if ref_accuracy - curr_accuracy > 0.1:  # if yes, trigger training workflow.
        logger.info(
            'Model in Production has poor performance, starting training workflow'
        )
        start_train()
    else:
        logger.info('Model Performance still up to par')
    logger.info('Done.')


@flow
def batch_analyze():
    upload_target.submit("target.csv")
    path = os.getenv('DATA_PATH')
    ref_data = load_reference_data(path)
    data = fetch_data.submit()
    result_future = run_evidently.submit(ref_data, data, wait_for=[data])
    result = result_future.result()
    save_report.submit(result, wait_for=[result_future])
    save_html_report.submit(result, wait_for=[result_future])
    check_model_performance(result)


if __name__ == '__main__':
    batch_analyze()
