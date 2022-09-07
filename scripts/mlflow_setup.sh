#!/usr/bin/env bash

KEY=modules/ec2/webserver_key.pem

if ! [ -e  $KEY ]; then
    aws ec2 create-key-pair --key-name webserver_key | jq -r ".KeyMaterial" > modules/ec2/webserver_key.pem && \
    chmod 400 $KEY
fi
terraform init -backend-config="key=mlops-final-prod.tfstate" -reconfigure && \
    terraform apply -target=module.mlflow_server -var-file=vars/prod.tfvars
echo "MLFLOW_TRACKING_URI = $(terraform output mlflow_server | sed -e 's/^"//' -e 's/"$//')" >> ../.env
