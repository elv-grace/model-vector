#!/bin/bash

set -e

git submodule update --init --recursive

# blip2-itm-vit-g is pulled from the HuggingFace hub at runtime into a mounted 
# HF cache, so there are no baked-in weights to sync before building.
exec buildscripts/build_container.bash -t "blip2-frame-vectors:${IMAGE_TAG:-latest}" . -f Containerfile
