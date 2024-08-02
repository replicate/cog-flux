#!/bin/bash

cat cog.yaml.template > cog.yaml
cat model-cog-configs/$1.yaml >> cog.yaml