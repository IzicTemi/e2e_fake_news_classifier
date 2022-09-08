#!/usr/bin/env bash

export ACCOUNT_ID=$(aws sts get-caller-identity --query "Account" --output text)
aws ecr get-login-password --region ${AWS_DEFAULT_REGION} | docker login --username AWS --password-stdin ${ACCOUNT_ID}.dkr.ecr.${AWS_DEFAULT_REGION}.amazonaws.com
docker tag ${LOCAL_IMAGE_NAME} ${ACCOUNT_ID}.dkr.ecr.${AWS_DEFAULT_REGION}.amazonaws.com/${LOCAL_IMAGE_NAME}
docker push ${ACCOUNT_ID}.dkr.ecr.${AWS_DEFAULT_REGION}.amazonaws.com/${LOCAL_IMAGE_NAME}
STATE=$(aws lambda get-function --function-name ${LAMBDA_FUNCTION} --region ${AWS_DEFAULT_REGION} --query 'Configuration.LastUpdateStatus' --output text)
    while [[ "$STATE" == "InProgress" ]]
    do
        echo "sleep 5sec ...."
        sleep 5s
        STATE=$(aws lambda get-function --function-name ${LAMBDA_FUNCTION} --region ${AWS_DEFAULT_REGION} --query 'Configuration.LastUpdateStatus' --output text)
        echo $STATE
    done

aws lambda update-function-code --function-name ${LAMBDA_FUNCTION} --image-uri ${ACCOUNT_ID}.dkr.ecr.${AWS_DEFAULT_REGION}.amazonaws.com/${LOCAL_IMAGE_NAME}

STATE=$(aws lambda get-function --function-name ${LAMBDA_FUNCTION} --region ${AWS_DEFAULT_REGION} --query 'Configuration.LastUpdateStatus' --output text)
    while [[ "$STATE" == "InProgress" ]]
    do
        echo "sleep 5sec ...."
        sleep 5s
        STATE=$(aws lambda get-function --function-name ${LAMBDA_FUNCTION} --region ${AWS_DEFAULT_REGION} --query 'Configuration.LastUpdateStatus' --output text)
        echo $STATE
    done

echo "Successfully deployed Model. To test make a POST request to $(cd infrastructure && terraform output lambda_rest_api_url)"
