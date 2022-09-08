LOCAL_TAG:=$(shell date +"%Y-%m-%d-%H")
LOCAL_IMAGE_NAME:=${ECR_REPO_NAME}:${LOCAL_TAG}
SHELL:=/bin/bash

test:
	pytest tests/

quality_checks:
	isort .
	black .
	pylint --recursive=y .

build: quality_checks test
	cd web_service && bash ./run.sh

integration_test: build
	LOCAL_IMAGE_NAME=${LOCAL_IMAGE_NAME} bash integration-test/run.sh

publish: integration_test create_key
	cd infrastructure && terraform apply -var-file=vars/prod.tfvars
	LAMBDA_FUNCTION=$(shell cd infrastructure && terraform output lambda_function)\
    LOCAL_IMAGE_NAME=${LOCAL_IMAGE_NAME} bash scripts/publish.sh

setup:
	pip install -U pip
	pip install pipenv
	pipenv install --dev
	pip install tf-nightly -q
	pre-commit install

create_bucket: create_key
	cd infrastructure && terraform init -backend-config="key=mlops-final-prod.tfstate" -reconfigure && terraform apply -target=module.s3_bucket -var-file=vars/prod.tfvars

setup_tf_vars:
	sed -i "s/model_bucket.*/model_bucket = \"${MODEL_BUCKET}\"/g" infrastructure/vars/prod.tfvars && \
	sed -i "s/ecr_repo_name.*/ecr_repo_name = \"${ECR_REPO_NAME}\"/g" infrastructure/vars/prod.tfvars && \
	sed -i "s/project_id.*/project_id = \"${PROJECT_ID}\"/g" infrastructure/vars/prod.tfvars && \
	sed -i "5s/bucket.*/bucket = \"${TFSTATE_BUCKET}\"/g" infrastructure/main.tf

mlflow_server: create_key
	cd infrastructure && bash ../scripts/mlflow_server.sh

create_key:
	sudo apt install -y jq
	cd infrastructure && bash ../scripts/mlflow_setup.sh

destroy: create_key
	cd infrastructure && terraform destroy -var-file=vars/prod.tfvars
	aws ec2 delete-key-pair --key-name webserver_key
	[ -f infrastructure/modules/ec2/webserver_key.pem ] && rm -f infrastructure/modules/ec2/webserver_key.pem
