#!/bin/bash

cog push r8.im/replicate/flux-schnell-setup
date +"%Y-%m-%d %H:%M:%S" > the_time.txt
yolo push -e FLUX_MODEL=FLUX_DEV --base r8.im/replicate/flux-schnell-setup --dest r8.im/replicate/flux-dev-setup the_time.txt