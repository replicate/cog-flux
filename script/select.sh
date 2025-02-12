#!/bin/bash

if [[ $1 == --list ]]; then
    ls model-cog-configs | sed 's/.yaml//'
    exit 0
fi

if [ -z $1 ]; then
    echo "Usage:"
    echo "  ./script/select.sh <model>"
    echo
    echo "To see all models: ./script/select.sh --list"
    exit 1
fi

yq eval-all 'select(fileIndex == 0) *+ select(fileIndex == 1)' cog.yaml.template "model-cog-configs/$1.yaml" > cog.yaml

cp safe-push-configs/$1.yaml cog-safe-push.yaml
