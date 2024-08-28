#!/bin/bash

if [ $# -ne 2 ]; then
    echo "Usage: $0 <model_name> <environment>"
    echo "Environment should be either 'test' or 'prod'"
    echo "Model name should be either 'dev' or 'schnell'"
    exit 1
fi

ENVIRONMENT="$2"

# Validate environment argument
if [ "$ENVIRONMENT" != "test" ] && [ "$ENVIRONMENT" != "prod" ]; then
    echo "Invalid environment. Please use 'test' or 'prod'."
    exit 1
fi

# Conditional cog push based on environment
if [ "$ENVIRONMENT" == "test" ]; then
    MODEL_NAME= "replicate-internal/flux-$MODEL_NAME"
elif [ "$ENVIRONMENT" == "prod" ]; then
    MODEL_NAME= "replicate/flux-$MODEL_NAME-internal-model"
fi

echo "Running tests on $MODEL_NAME"

pytest -vv integration-tests/
