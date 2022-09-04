#!/usr/bin/env bash

export KAGGLE_USERNAME="izictemi"
export KAGGLE_KEY="3e6c0e9c5c45fc5b1e2137c5175848db"
export DATA_PATH="data"
export ARTIFACT_LOC="s3://my-mlflow-models-bucket"
export MLFLOW_TRACKING_URI="https://dagshub.com/IzicTemi/mlops_zoomcamp_final_project.mlflow"
export MLFLOW_TRACKING_USERNAME="IzicTemi"
export MLFLOW_TRACKING_PASSWORD="61acfdb3d6cc1fe5b9a65af4b498dee79718ec52"
export AWS_SECRET_ACCESS_KEY="KReV+fwn22KBJ0WZGwL3KSSgAl+HDkhgxwvtTdbb"
export AWS_ACCESS_KEY_ID="AKIAU7XCE744L6KEMWW4"
export AWS_DEFAULT_REGION="us-east-1"

touch .env
echo "KAGGLE_USERNAME = $KAGGLE_USERNAME" > .env
echo "KAGGLE_KEY = $KAGGLE_KEY" >> .env
echo "DATA_PATH = $DATA_PATH" >> .env
echo "MLFLOW_TRACKING_URI = $MLFLOW_TRACKING_URI" >> .env
echo "MLFLOW_TRACKING_USERNAME = $MLFLOW_TRACKING_USERNAME" >> .env
echo "MLFLOW_TRACKING_PASSWORD = $MLFLOW_TRACKING_PASSWORD" >> .env
echo "AWS_SECRET_ACCESS_KEY = $AWS_SECRET_ACCESS_KEY" >> .env
echo "AWS_ACCESS_KEY_ID = $AWS_ACCESS_KEY_ID" >> .env
echo "AWS_DEFAULT_REGION = $AWS_DEFAULT_REGION" >> .env
