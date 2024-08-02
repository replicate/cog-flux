# FLUX (in cog!)

This is a repository for running flux-dev and flux-schnell within a cog container. 

## How to use this repo

### Selecting a model

run `script/select.sh (dev,schnell)` and that'll create a cog.yaml configured for the appropriate model.

### Pushing a model

run `script/push.sh (dev,schnell)` to push the model to Replicate. Note that these models are currently configured
to push to replicate internal repos. 

To push all models, run `script/cog-push-all.sh`