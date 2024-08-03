#!/bin/bash

./script/select.sh "$1"
if [ $? -ne 0 ]; then
    echo "Couldn't select a model, double check you're passing a valid name."
    exit 1
fi

v=$(cog --version)

# async cog?
if [[ $v == *"0.10.0"* ]]; then
    echo "Async cog found, pushing model"
else
    echo "Nope! switch to async cog and rebuild"
    exit -1
fi

cog push r8.im/replicate-internal/flux-$1