LOCAL_TAG:=$(shell date +"%Y-%m-%d-%H")
LOCAL_IMAGE_NAME:=${ECR_REPO_NAME}-${PROJECT_ID}:${LOCAL_TAG}
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

publish: integration_test
	cd infrastructure && terraform apply -var-file=vars/prod.tfvars
	LAMBDA_FUNCTION=$(shell cd infrastructure && terraform output lambda_function)\
    LOCAL_IMAGE_NAME=${LOCAL_IMAGE_NAME} bash scripts/publish.sh

setup:
	pip install -U pip
	pipenv install --dev
	pip install tf-nightly
	pre-commit install

create-bucket:
	cd infrastructure && terraform init && terraform apply -target=module.s3_bucket -var-file=vars/prod.tfvars
