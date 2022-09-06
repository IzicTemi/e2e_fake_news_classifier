Pre-requisite

Setup an AWS account

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


Install Terraform

https://learn.hashicorp.com/tutorials/terraform/install-cli


## Steps to Take

Edit scripts/set_env.sh

Set Environment Variables
```
source scripts/set_env.sh
```
Create S3 Bucket to hold Terraform State

```
aws s3api create-bucket --bucket $TFSTATE_BUCKET \
--region $AWS_DEFAULT_REGION
```

Set Terraform variables
```
sed -i "s/model_bucket.*/model_bucket = $MODEL_BUCKET/g" infrastructure/vars/prod.tfvars

sed -i "s/ecr_repo_name.*/ecr_repo_name = $ECR_REPO_NAME/g" infrastructure/vars/prod.tfvars

sed -i "s/project_id.*/project_id = $PROJECT_ID/g" infrastructure/vars/prod.tfvars

sed -i "5s/bucket.*/bucket = \"$TFSTATE_BUCKET\"/g" infrastructure/main.tf
```

If AWS default region not us-east-1, run
```
find . -type f -exec sed -i "s/us-east-1/$AWS_DEFAULT_REGION/g" {} \;
```
## Initial Setup
```
make setup
```

Start shell
```
pipenv shell
```

Create Bucket to store Mlflow artifacts

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
```
python train.py --n_evals 2 --epochs 3

python train.py --n_evals 1 --epochs 1
```

Tests
```
make test
make integration_test
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
