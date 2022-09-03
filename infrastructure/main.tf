# Make sure to create state bucket beforehand
terraform {
  required_version = ">= 1.0"
  backend "s3" {
    bucket  = "my-tf-state-mlops-zoomcamp"
    key     = "mlops-final-prod.tfstate"
    region  = var.region
    encrypt = true
  }
}

provider "aws" {
  region = var.region
}

***REMOVED*** "aws_caller_identity" "current_identity" {}

locals {
  account_id = ***REMOVED***.aws_caller_identity.current_identity.account_id
}

# model bucket
module "s3_bucket" {
  source = "./modules/s3"
  bucket_name = "${var.model_bucket}-${var.project_id}"
}

# image registry
module "ecr_image" {
   source = "./modules/ecr"
   ecr_repo_name = "${var.ecr_repo_name}-${var.project_id}"
   account_id = local.account_id
   lambda_function_local_path = var.lambda_function_local_path
   docker_image_local_path = var.docker_image_local_path
}

module "lambda_function" {
  source = "./modules/lambda"
  image_uri = module.ecr_image.image_uri
  lambda_function_name = "${var.lambda_function_name}-${var.project_id}"
  model_bucket = module.s3_bucket.name
  account_id = local.account_id
}

# For CI/CD
output "lambda_function" {
  value     = "${var.lambda_function_name}-${var.project_id}"
}

output "model_bucket" {
  value = module.s3_bucket.name
}

output "ecr_repo" {
  value = "${var.ecr_repo_name}-${var.project_id}"
}

output "lambda_rest_api_url" {
  value = module.lambda_function.base_url
}