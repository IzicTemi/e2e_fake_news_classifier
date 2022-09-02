resource "aws_lambda_function" "pred_lambda" {
  function_name = var.lambda_function_name
  # This can also be any base image to bootstrap the lambda config, unrelated to your Inference service on ECR
  # which would be anyway updated regularly via a CI/CD pipeline
  image_uri = var.image_uri   # required-argument
  package_type = "Image"
  role          = aws_iam_role.iam_lambda.arn
  tracing_config {
    mode = "Active"
  }
  // This step is optional (environment)
  environment {
    variables = {
      MODEL_BUCKET = var.model_bucket
    }
  }
  timeout = 180
  memory_size = 4000
}

# Lambda Invoke & Event Source Mapping:

resource "aws_api_gateway_rest_api" "lambda-api" {
  name          = "serverless_lambda_gw"
}

resource "aws_api_gateway_resource" "proxypred" {
   rest_api_id = aws_api_gateway_rest_api.lambda-api.id
   parent_id   = aws_api_gateway_rest_api.lambda-api.root_resource_id
   path_part   = "classify"
}

resource "aws_api_gateway_method" "methodproxy" {
   rest_api_id   = aws_api_gateway_rest_api.lambda-api.id
   resource_id   = aws_api_gateway_resource.proxypred.id
   http_method   = "POST"
   authorization = "NONE"
 }

resource "aws_api_gateway_integration" "apilambda" {
   rest_api_id = aws_api_gateway_rest_api.lambda-api.id
   resource_id = aws_api_gateway_method.methodproxy.resource_id
   http_method = aws_api_gateway_method.methodproxy.http_method

   integration_http_method = "POST"
   type                    = "AWS"
   uri                     = aws_lambda_function.pred_lambda.invoke_arn
   timeout_milliseconds = 29000
 }

resource "aws_api_gateway_method_response" "response_200" {
  rest_api_id = aws_api_gateway_rest_api.lambda-api.id
  resource_id = aws_api_gateway_resource.proxypred.id
  http_method = aws_api_gateway_method.methodproxy.http_method
  status_code = "200"
}

 resource "aws_api_gateway_integration_response" "MyDemoIntegrationResponse" {
  rest_api_id = aws_api_gateway_rest_api.lambda-api.id
  resource_id = aws_api_gateway_resource.proxypred.id
  http_method = aws_api_gateway_method.methodproxy.http_method
  status_code = aws_api_gateway_method_response.response_200.status_code
}

 resource "aws_lambda_permission" "apigw_lambda" {
  statement_id  = "AllowExecutionFromAnyWhere"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.pred_lambda.function_name
  principal     = "apigateway.amazonaws.com"

  # More: http://docs.aws.amazon.com/apigateway/latest/developerguide/api-gateway-control-access-using-iam-policies-to-invoke-api.html
  source_arn = "arn:aws:execute-api:${var.region}:${var.account_id}:${aws_api_gateway_rest_api.lambda-api.id}/*/${aws_api_gateway_method.methodproxy.http_method}${aws_api_gateway_resource.proxypred.path}"
}

resource "aws_api_gateway_deployment" "apideploy" {
  depends_on = [
    aws_api_gateway_integration.apilambda
  ]

  rest_api_id = aws_api_gateway_rest_api.lambda-api.id

  triggers = {
  redeployment = aws_api_gateway_resource.proxypred.path
}
  lifecycle {
    create_before_destroy = true
  }
}

resource "aws_api_gateway_stage" "test" {
  deployment_id = aws_api_gateway_deployment.apideploy.id
  rest_api_id   = aws_api_gateway_rest_api.lambda-api.id
  stage_name    = "testing"
}

# IAM for api

resource "aws_api_gateway_rest_api_policy" "test" {
  rest_api_id = aws_api_gateway_rest_api.lambda-api.id

  policy = <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": "*",
      "Action": [
        "execute-api:Invoke"           
      ],
      "Resource": [
        "arn:aws:execute-api:${var.region}:${var.account_id}:${aws_api_gateway_rest_api.lambda-api.id}/*/${aws_api_gateway_method.methodproxy.http_method}${aws_api_gateway_resource.proxypred.path}"
      ]
    }
  ]
} 
EOF
}

output "base_url" {
  value = "${aws_api_gateway_stage.test.invoke_url}${aws_api_gateway_resource.proxypred.path}"
}