variable "region" {
  description = "AWS region to create resources"
}

variable "project_id" {
  description = "project_id"
  default = "mlops-zoomcamp"
}

variable "model_bucket" {
  description = "s3_bucket"
}

variable "ecr_repo_name" {
  description = ""
}

variable "lambda_function_name" {
  description = ""
}
