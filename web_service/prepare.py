import os

import mlflow
from mlflow.tracking import MlflowClient


def main(MLFLOW_TRACKING_URI, model_name):
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    prod_version = client.get_latest_versions(name=model_name, stages=["Production"])[0]
    run_id = prod_version.run_id

    mlflow.artifacts.download_artifacts(run_id=run_id, dst_path='./artifact')


if __name__ == '__main__':
    MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI')
    model_name = os.getenv('MODEL_NAME')
    main(MLFLOW_TRACKING_URI, model_name)
