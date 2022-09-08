#!/bin/sh

sudo yum -y -q update
export PATH=$PATH:~/.local/bin
pip3 install mlflow boto3 psycopg2-binary -q
nohup mlflow server -h 0.0.0.0 -p 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root s3://$ARTIFACT_LOC > foo.out 2> foo.err < /dev/null &
sleep 5
exit
