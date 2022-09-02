LOCAL_TAG:=$(shell date +"%Y-%m-%d-%H")
LOCAL_IMAGE_NAME:=fake-news-classifier:${LOCAL_TAG}
SHELL:=/bin/bash

test:
	pytest tests/

quality_checks:
	isort .
	black .
	pylint --recursive=y .

build: quality_checks test
	docker build -t ${LOCAL_IMAGE_NAME} .

integration_test: build
	LOCAL_IMAGE_NAME=${LOCAL_IMAGE_NAME} bash integraton-test/run.sh

publish: build integration_test
	LOCAL_IMAGE_NAME=${LOCAL_IMAGE_NAME} bash scripts/publish.sh

setup:
	pipenv install --dev
	pre-commit install

create-bucket:
	cd infrastructure && terraform apply -target=module.s3_bucket -var-file=vars/prod.tfvars

