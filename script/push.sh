#!/bin/bash

# Check if both arguments are provided
if [ $# -ne 2 ]; then
    echo "Usage: $0 <model_name> <environment>"
    echo "Environment should be either 'test' or 'prod'"
    exit 1
fi

MODEL_NAME="$1"
ENVIRONMENT="$2"

# Validate environment argument
if [ "$ENVIRONMENT" != "test" ] && [ "$ENVIRONMENT" != "prod" ]; then
    echo "Invalid environment. Please use 'test' or 'prod'."
    exit 1
fi

./script/select.sh "$MODEL_NAME"

if [ $? -ne 0 ]; then
    echo "Couldn't select a model, double check you're passing a valid name."
    exit 1
fi

v=$(cog --version)

# async cog?
if [[ $v == *"0.9."* ]]; then
    echo "Sync cog found, pushing model"
else
    echo "Nope! switch to sync cog and rebuild"
    exit -1
fi

# Conditional cog push based on environment
if [ "$ENVIRONMENT" == "test" ]; then
    echo "Pushing to test environment"
    cog push r8.im/replicate-internal/flux-$MODEL_NAME 
elif [ "$ENVIRONMENT" == "prod" ]; then
    echo "Pushing to production environment"
    cog push r8.im/replicate/flux-$MODEL_NAME-internal-model 
fi