variable "model_bucket" {
  description = "Name of the bucket"
}

variable "lambda_function_name" {
  description = "Name of the lambda function"
}

variable "image_uri" {
  description = "ECR image uri"
}
variable "region" {
    type        = string
    description = "region"
}

variable "account_id" {
}
