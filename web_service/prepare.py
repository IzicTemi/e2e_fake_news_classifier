import os

import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

MLFLOW_TRACKING_URI = os.environ['MLFLOW_TRACKING_URI']
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

model_name = "fake-news-detect-1"

model_versions = client.search_model_versions(f"name='{model_name}'")
run_id = dict(model_versions[0])['run_id']

mlflow.artifacts.download_artifacts(run_id=run_id, dst_path='./artifact')
