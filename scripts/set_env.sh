#!/usr/bin/env bash

export MLFLOW_TRACKING_PASSWORD=***REMOVED***
export MLFLOW_TRACKING_URI=***REMOVED***
export MLFLOW_TRACKING_USERNAME=***REMOVED***
export AWS_SECRET_ACCESS_KEY=***REMOVED***
export AWS_ACCESS_KEY_ID=***REMOVED***
export KAGGLE_USERNAME=***REMOVED***
export KAGGLE_KEY=***REMOVED***
export DATA_PATH=***REMOVED***
export ARTIFACT_LOC=***REMOVED***


touch ../.env
echo "KAGGLE_USERNAME = $KAGGLE_USERNAME" > ../.env
echo "KAGGLE_KEY = $KAGGLE_KEY" >> ../.env
echo "DATA_PATH = $DATA_PATH" >> ../.env
