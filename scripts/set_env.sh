#!/usr/bin/env bash

export MLFLOW_TRACKING_PASSWORD=
export MLFLOW_TRACKING_URI=
export MLFLOW_TRACKING_USERNAME=
export AWS_SECRET_ACCESS_KEY=
export AWS_ACCESS_KEY_ID=
export KAGGLE_USERNAME=
export KAGGLE_KEY=
export DATA_PATH=
export ARTIFACT_LOC=


touch ../.env
echo "KAGGLE_USERNAME = $KAGGLE_USERNAME" > ../.env
echo "KAGGLE_KEY = $KAGGLE_KEY" >> ../.env
echo "DATA_PATH = $DATA_PATH" >> ../.env
