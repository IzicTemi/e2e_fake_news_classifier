#!/usr/bin/env bash

KEY=modules/ec2/webserver_key.pem

if ! [ -e  $KEY ]; then
    aws ec2 create-key-pair --key-name webserver_key | jq -r ".KeyMaterial" > modules/ec2/webserver_key.pem && \
    chmod 400 $KEY
fi
