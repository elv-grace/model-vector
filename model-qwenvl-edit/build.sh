#!/bin/bash

set -e

# The weights live locally under embedding/Qwen3-VL-Embedding-8B/ (gitignored) and
# are baked into the image by the Containerfile. Verify the weights are present before building.
# TODO: mount a cache folder at the path where huggingface writes its weights for Qwen3-VL-Embedding-8B 
# so weights can be loaded and reused at runtime and not redownloaded each time (change to shared-source rsync like model-shot/-celeb).
if [ ! -d "embedding/Qwen3-VL-Embedding-8B" ] || [ -z "$(ls -A embedding/Qwen3-VL-Embedding-8B 2>/dev/null)" ]; then
    echo "error: embedding/Qwen3-VL-Embedding-8B/ is missing or empty; place the model weights there before building." >&2
    exit 1
fi

exec buildscripts/build_container.bash -t "qwenvl-embedding:${IMAGE_TAG:-latest}" . -f Containerfile
