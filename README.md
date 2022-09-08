# End to End Fake News Classifier

This repository contains an implementation of an end to end fake news classifier.

It is the final capstone project for
[MLOps Zoomcamp course](https://github.com/DataTalksClub/mlops-zoomcamp) from [DataTalks.Club](https://datatalks.club).

## About the Project

Misinformation is wide spread. The aim of the project is to train and deploy a model to detect the presence of fake claims in articles.

Emphasis is largely placed on the MLOps pipeline.

## About the Data

Data source: [Fake and real news dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

This data consists of about 40000 articles consisting of fake and real news. The data consists of two separate datasets - one for each news category with each dataset containing around 20000 articles.

## Project Solution and Architecture

Tools used include:
1. [Terraform](https://www.terraform.io) is the Infrastructure as Code (IaC) tool used for creating resources.
2. [MLflow](https://www.mlflow.org) for experiment tracking.
3. [Docker](https://www.docker.com) for containerization.
4. [Prefect 2.0](https://www.prefect.io/opensource/v2/) for workflow orchestration.
5. [AWS S3 Lambda](https://aws.amazon.com/s3/) for cloud deployment and inference.
6. [Flask](https://flask.palletsprojects.com/en/2.2.x/) for local deployment and inference.
7. [Github Actions](https://github.com/features/actions) for Continuous Integration and Continuous Delivery.

### Machine Learning Model

The model builds on ideas from Madhav Mathur's [notebook](https://www.kaggle.com/code/madz2000/nlp-using-glove-embeddings-99-87-accuracy).

Words are represented using GloVe Embeddings which is a word vector technique. GloVe incorporates global statistics (word co-occurrence) to obtain word vectors. More info about GloVe [here](https://towardsdatascience.com/light-on-math-ml-intuitive-guide-to-understanding-glove-embeddings-b13b4f19c010?gi=b9a4fd6dc8ff).

An LSTM model with 5 layers was trained using Tensorflow and Keras.

### Pre-requisite

Optional - Create a VM with about 8gbs of RAM. This would allow for fast training, downloading the fairly large dataset (~2gb), pulling and pushing of required docker containers.

Set up an AWS account.

Set up a Kaggle account for getting the data.

Install Python 3.9
```
wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.12.0-Linux-x86_64.sh
chmod +x Miniconda3-py39_4.12.0-Linux-x86_64.sh
./Miniconda3-py39_4.12.0-Linux-x86_64.sh
rm Miniconda3-py39_4.12.0-Linux-x86_64.sh
```

Install aws-cli
```
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
sudo apt install unzip
unzip awscliv2.zip
sudo ./aws/install
rm -r awscliv2.zip aws/
```

Create AWS user with administrator access. Note the AWS_SECRET_ACCESS_KEY and AWS_ACCESS_KEY_ID.

https://docs.aws.amazon.com/IAM/latest/UserGuide/id_users_create.html

Install docker

https://docs.docker.com/desktop/install/linux-install/

Allow docker to run without sudo

https://docs.docker.com/engine/install/linux-postinstall/

Install docker-compose
```
sudo apt install docker-compose
```

Install Terraform

https://learn.hashicorp.com/tutorials/terraform/install-cli

### Getting Started

If interested in testing automated deploy capabilities using Github Actions, fork the repository and clone fork to local machine.

**OR**

To test locally or manually deploy, clone the repository to your local machine.
```
git clone https://github.com/IzicTemi/mlops_zoomcamp_final_project.git
```

### Preparing your workspace


1. Set Environment Variables

Edit [set_env.sh](scripts/set_env.sh) in [scripts](scripts/) folder.

 - Get KAGGLE_USERNAME and KAGGLE_KEY following the instructions [here](https://www.kaggle.com/docs/api#:~:text=is%20%24PYTHON_HOME/Scripts.-,Authentication,-In%20order%20to).

 - DATA_PATH is path to store data. Preferrably "data".

 - MODEL_BUCKET is the intended name of s3 bucket to store MLflow artifacts.

 - PROJECT_ID is the tag to add to created resources to ensure uniqueness.

 - MLFLOW_TRACKING_URI is the tracking server url. Default is http://127.0.0.1:5000 for local MLflow setup. Leave empty if you want to setup Mlfow on AWS ec2 instance.

 - TFSTATE_BUCKET is the intended name of s3 bucket to store Terraform State files.

 - AWS_SECRET_ACCESS_KEY and AWS_ACCESS_KEY_ID from user created above.

 - AWS_DEFAULT_REGION is the default region for resources to be created.

 - ECR_REPO_NAME is the intended name of ECR registry to store docker images.

 - MODEL_NAME is the name to which to register the trained models.

 Optional

 - MLFLOW_TRACKING_USERNAME and MLFLOW_TRACKING_PASSWORD if using an authenticated MLflow server.

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
4. Optional - Set up MLflow Server on ec2 instance

Manually setup MLflow on an ec2 instance by following instructions [here](https://github.com/DataTalksClub/mlops-zoomcamp/blob/main/02-experiment-tracking/mlflow_on_aws.md).

**OR**

Run
```
make mlflow_server
```
- The above command creates a free tier eligible t2.micro ec2 instance and installs MLflow on it using Terraform.
- It also creates a key pair called webserver_key and downloads the private key to the [ec2 module](infrastructure/modules/ec2) folder in the infrastructure folder. This allows Terraform interact with the ec2 instance.
- An sqlite db is used as the backend store. In the future, a better implementation would be to use a managed RDS instance. This could be added later.

5. Optional - If AWS default region not us-east-1, run:
```
find . -type f -exec sed -i "s/us-east-1/$AWS_DEFAULT_REGION/g" {} \;
```

### Implementing the Solution

1. Install dependencies and setup environment
```
make setup
```
- The above command install pipenv which in turn sets up the virtual environment.
- It also installs the pre commit hooks.

2. Start virtual environment
```
pipenv shell
```

3. Create S3 bucket to store MLflow artifacts
```
make create_bucket
```

4. Get dataset from Kaggle
```
python get_data.py
```

5. Optional - If running locally, start MLflow server
```
mlflow server --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root $ARTIFACT_LOC --serve-artifacts
```

6. Train the model

The model training process performs a hyperparameter search to get best parameters. This could take very long and is memory intensive. If interested in the full training process, run:
```
python train.py
```

For testing purposes, set a small number of optimization trials and lower the number of epochs required to train the model.
```
python train.py --n_evals 2 --epochs 3
```

7. Workflow Orchestration with Prefect

To automate getting the data and training the model on a schedule run:
```
python prefect_deploy.py

prefect agent start  --work-queue "main"
```
- The above script uses [Prefect](https://www.prefect.io/opensource/v2/) to automate the deployment. Using a Cron Scheduler currently set to run by 00:00 every Monday, the agent looks for work and runs it at the appointed time.
- To change the schedule, edit the [prefect_deploy.py](prefect_deploy.py) file and change the Cron schedule.
- To view the scheduled deployments, run:
```
prefect orion start
```

8. Deploying the Model
<ol type="a">

<li> Deploy web service locally using Flask</li>

```
cd web_service_local

./run.sh
```
- To make inferences, make a POST request to http://127.0.0.1:9696/classify
- The content of the POST request should be of the format:
```
{
    'text': text
}
```
**OR**

Edit and run [test.py](web_service_local/test.py) in [web_service_local](web_service_local/) folder
```
python web_service_local/test.py
```

<li> Manually deploy web service to AWS Lambda </li>

```
make publish
```
- The above command uses Terraform to deploy the model to AWS Lambda and exposes it using an API gateway endpoint.
- The scripts outputs the endpoint of the Lambda function.
- To make inferences, make a POST request to the output url.
- The content of the POST request should be of the format:
```
{
    'text': text
}
```
**OR**

Edit and run [test.py](web_service/test.py) in [web_service](web_service/) folder
```
python web_service/test.py
```
- If you get a {'message': 'Endpoint request timed out'} error, retry the request; the initial model loading takes time.

</ol>

### Tests
```
make test
make integration_test
```
## Continuous Integration and Deployment

1. Fork repository and clone fork to local machine.

2. Switch to branch test-branch
```
git checkout test-branch
```

3. Perform all steps in [Preparing your workspace](#preparing-your-workspace) above and steps 1 - 7 from [Implementing the Solution](#implementing-the-solution).

4. Add Github Secrets to the forked repository as shown in the image below:

![Alt text](images/github_secrets.png?raw=true "Title")
- Copy the output of the command below and set as the value SSH_PRIVATE_KEY. This allows terraform interact with the MLflow Server.
```
cat infrastructure/modules/ec2/webserver_key.pem
```

5. Edit [ci-tests.yaml](.github/workflows/ci-tests.yaml) and [cd-deploy.yml](.github/workflows/cd-deploy.yml) in [.github/workflows](.github/workflows/) folder.

- Replace env variable MODEL_NAME in [ci-tests.yaml](.github/workflows/ci-tests.yaml#L14) and [cd-deploy.yml](.github/workflows/cd-deploy.yml#11)
- Replace env variable ECR_REPO_NAME in [ci-tests.yaml](.github/workflows/ci-tests.yaml#L15)

6. Commit and push changes to Github.

7. On branch develop of the forked repo, create a pull request to merge test-branch into develop.
- This triggers the Continuous Integration workflow which runs unit tests, integration test and validates the Terraform configuration

8. After all checks are completed, merge pull request into develop
- This triggers the Continuous Deployment workflow which applies the Terrafrom configuration and deploys the infrastructure.

## Destroy Infrastructure

On completing all the steps above, destroy all the setup infrastructure by running:
```
make destroy
```
**Note**: This destroys all created infrastructure including MLflow Server and bucket. To prevent destruction of the bucket, edit the [s3 module](infrastructure/modules/s3/main.tf#L3) in the Terraform configuration and set:
```
force_destroy = false
```
