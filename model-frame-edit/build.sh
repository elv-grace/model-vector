#!/bin/bash

set -e

# No local weights to sync: the Containerfile pulls Salesforce/blip2-itm-vit-g from the
# HuggingFace hub and bakes it into the image's HF cache at build time.
exec buildscripts/build_container.bash -t "blip2-frame:${IMAGE_TAG:-latest}" . -f Containerfile
