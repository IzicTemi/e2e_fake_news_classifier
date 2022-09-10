#!/usr/bin/env bash

if [[ -z "${GITHUB_ACTIONS}" ]]; then
  cd "$(dirname "$0")"
fi

if [ "${LOCAL_IMAGE_NAME}" == "" ]; then
    LOCAL_TAG=`date +"%Y-%m-%d-%H"`
    export LOCAL_IMAGE_NAME="${ECR_REPO_NAME}:${LOCAL_TAG}"
    echo "LOCAL_IMAGE_NAME is not set, building a new image with tag ${LOCAL_IMAGE_NAME}"
    docker build -t ${LOCAL_IMAGE_NAME} ../web_service_local \
    --build-arg MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI} \
    --build-arg MLFLOW_TRACKING_USERNAME=${MLFLOW_TRACKING_USERNAME} \
    --build-arg MLFLOW_TRACKING_PASSWORD=${MLFLOW_TRACKING_PASSWORD} \
    --build-arg AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} \
    --build-arg AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} \
    --build-arg MODEL_NAME=${MODEL_NAME}

else
    echo "no need to build image ${LOCAL_IMAGE_NAME}"
fi

docker-compose up --build -d
