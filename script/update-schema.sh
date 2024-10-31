#!/bin/bash

MODEL_NAME="$1"

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
echo "Pushing image to prod to update schema"
date +%s > the_time.txt
cog push r8.im/black-forest-labs/flux-$MODEL_NAME