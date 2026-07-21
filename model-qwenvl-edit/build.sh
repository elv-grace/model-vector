#!/bin/bash

set -e

# Qwen3-VL-Embedding-8B is pulled from the HuggingFace hub at runtime into a mounted HF cache 
# (see README "Weights" / "Run"). Weights are not baked into the image, so there is nothing to sync before building.
exec buildscripts/build_container.bash -t "qwenvl-embedding:${IMAGE_TAG:-latest}" . -f Containerfile
