# End to End Fake News Detector (Not done)

## About the project (Not done)


## About the data (Not done)

data source: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

## Project Solution and Architecture


### Pre-requisite

Setup an AWS account

Setup a Kaggle account for getting the data

Install Python 3.9
```
wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.12.0-Linux-x86_64.sh
chmod +x Miniconda3-py39_4.12.0-Linux-x86_64.sh
./Miniconda3-py39_4.12.0-Linux-x86_64.sh
```
Install aws-cli
```
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
sudo apt install unzip
unzip awscliv2.zip
sudo ./aws/install
rm -r awscliv2.zip aws/
```
Create aws user with administrator access

https://docs.aws.amazon.com/IAM/latest/UserGuide/id_users_create.html

Install docker

https://docs.docker.com/desktop/install/linux-install/

Install Terraform

https://learn.hashicorp.com/tutorials/terraform/install-cli

### Getting Started

If interested in testing automated deploy capabilities using Github Actions, fork the repository and clone fork to local machine

OR

To test locally or manually deploy, clone the repository to local machine
```
git clone https://github.com/IzicTemi/mlops_zoomcamp_final_project.git
```

### Preparing your workspace


1. Set Environment Variables

Edit scripts/set_env.sh

 - Get KAGGLE_USERNAME and KAGGLE_KEY following the instructions [here](https://www.kaggle.com/docs/api#:~:text=is%20%24PYTHON_HOME/Scripts.-,Authentication,-In%20order%20to)

- DATA_PATH is path to store data. Preferrably "data"

 - MODEL_BUCKET is the intended name of s3 bucket to store Mlflow artifacts

 - PROJECT_ID is the tag to add to created resources to ensure uniqueness

 - MLFLOW_TRACKING_URI is the tracking server url. Default is "http://127.0.0.1:5000" for local Mlflow setup

 - TFSTATE_BUCKET is the intended name of s3 bucket to store Terraform State files

 - AWS_SECRET_ACCESS_KEY and AWS_ACCESS_KEY_ID from user created above

 - AWS_DEFAULT_REGION is the default region for resources to be created

 - ECR_REPO_NAME is the intended name of ECR registry to store docker images

 - MODEL_NAME is the name to which to register the trained models

 Optional

 - MLFLOW_TRACKING_USERNAME and MLFLOW_TRACKING_PASSWORD if using an authenticated mlflow server

Run command:
```
source scripts/set_env.sh
```
2. Create S3 Bucket to store Terraform States

This can be done from the console or by running
```
aws s3api create-bucket --bucket $TFSTATE_BUCKET \
--region $AWS_DEFAULT_REGION
```

3. Set Terraform variables
```
make setup_tf_vars
```

4. Optional - If AWS default region not us-east-1, run
```
find . -type f -exec sed -i "s/us-east-1/$AWS_DEFAULT_REGION/g" {} \;
```

### Running the Solution

Install dependencies and setup environment
```
make setup
```

Start shell
```
pipenv shell
```

Create S3 bucket to store Mlflow artifacts
```
make create-bucket
```

Get dataset
```
python get_data.py
```

Start Mlflow Server
```
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root $ARTIFACT_LOC --serve-artifacts
```

Train the model

The model training process performs a hyperparameter search to get best parameters. This could take very long and is memory intensive. If interested in the full training process, run
```
python train.py
```

For testing purposes, set a small number of optimization trials and lower the number of epochs required to train the model
```
python train.py --n_evals 3 --epochs 5
```

Deploy web service locally using Flask
```
cd web_service_local
./run.sh
```

Manually deploy web service to AWS Lambda
```
make pulish
```

Tests
```
make test
make integration_test
```

Create a key-pair call webserver_key and copy into modules/ec2
Can be done from the console or run:
```
sudo apt install jq
aws ec2 create-key-pair --key-name webserver_key | jq -r ".KeyMaterial" > modules/ec2/webserver_key.pem
chmod 400 modules/ec2/webserver_key.pem
```
cd infrastructure && terraform apply -target=module.mlflow_server -var-file=vars/prod.tfvars
