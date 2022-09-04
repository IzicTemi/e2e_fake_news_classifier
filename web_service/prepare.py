import os

import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

MLFLOW_TRACKING_URI = os.environ['MLFLOW_TRACKING_URI']
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

model_name = "fake-news-detect"

prod_version = client.get_latest_versions(name=model_name, stages=["Production"])[0]
run_id = prod_version.run_id

mlflow.artifacts.download_artifacts(run_id=run_id, dst_path='./artifact')
