#!/usr/bin/env bash

if [ "${LOCAL_IMAGE_NAME}" == "" ] || [ "${LOCAL_IMAGE_NAME}" != fake-news-classifier* ]; then
    LOCAL_TAG=`date +"%Y-%m-%d-%H"`
    export LOCAL_IMAGE_NAME="fake-news-classifier:${LOCAL_TAG}"
    echo "LOCAL_IMAGE_NAME is not set, building a new image with tag ${LOCAL_IMAGE_NAME}"
    docker build -t ${LOCAL_IMAGE_NAME} . \
    --build-arg MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI} \
    --build-arg MLFLOW_TRACKING_USERNAME=${MLFLOW_TRACKING_USERNAME} \
    --build-arg MLFLOW_TRACKING_PASSWORD=${MLFLOW_TRACKING_PASSWORD} \
    --build-arg AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} \
    --build-arg AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}

else
    echo "no need to build image ${LOCAL_IMAGE_NAME}"
fi

docker run -it --rm -p 9696:9696 ${LOCAL_IMAGE_NAME}
