# FLUX (in cog!)

This is a repository for running flux-dev and flux-schnell within a cog container. 

## How to use this repo

### Selecting a model

run `script/select.sh (dev,schnell)` and that'll create a cog.yaml configured for the appropriate model.

### Pushing a model

run `script/push.sh (dev,schnell) (test, prod)` to push the model to Replicate. 

To push all models, run `script/prod-deploy-all.sh`. Note that after doing this you'll still need to manually go in and update deployments. 
