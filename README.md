Initial Setup

make setup
make create-bucket

Edit set_env
source set_env.sh

Add option to train only once
python train.py --n_evals 1 --epochs 3

cd integration-test
docker build -t $LOCAL_IMAGE_NAME ../web_service \
    --build-arg MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI \
    --build-arg MLFLOW_TRACKING_USERNAME=$MLFLOW_TRACKING_USERNAME \
    --build-arg MLFLOW_TRACKING_PASSWORD=$MLFLOW_TRACKING_PASSWORD \
    --build-arg AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
    --build-arg AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID

pip install tf-nightly

Local image name in makefile

FIrst run
source scripts/set_env.sh
