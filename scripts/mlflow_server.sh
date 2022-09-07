#!/usr/bin/env bash

terraform init -backend-config="key=mlops-final-prod.tfstate" -reconfigure && \
    terraform apply -target=module.mlflow_server -var-file=vars/prod.tfvars
echo "MLFLOW_TRACKING_URI = $(terraform output mlflow_server | sed -e 's/^"//' -e 's/"$//')" >> ../.env
