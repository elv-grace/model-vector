#!/bin/bash

set -e

# blip2-itm-vit-g is pulled from the HuggingFace hub at runtime into a mounted 
# HF cache (README "Weights" / "Run"), so there are no baked-in weights to sync before building.
exec buildscripts/build_container.bash -t "blip2-frame:${IMAGE_TAG:-latest}" . -f Containerfile
