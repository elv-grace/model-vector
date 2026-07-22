#!/bin/bash
#
# Local smoke test for the qwen3vl-embedding-video-vector container.
#
# Thin, editable wrapper (mirrors buildscripts/testers/test-model.sh). All runtime
# tunables are RuntimeConfig fields in run.py, injected as one --params JSON string:
#
#   ./test.sh                                          # defaults: fps=1, max_frames=64, max_length=8192, whole-video
#   FPS=5 SEGMENT_LENGTH_S=30 ./test.sh                # override any subset of or all four parameters
#   MAX_FRAMES=32 MAX_LENGTH=4096 ./test.sh             (memory/context)
#   FPS=5 MAX_FRAMES=192 MAX_LENGTH=24576 SEGMENT_LENGTH_S=30 ./test.sh
#
# File paths are fed on STDIN (run_default reads stdin, not argv), --output-path is
# required, and the hf_cache named volume is mounted at HF_HOME (=/root/.cache, set in
# the Containerfile) so the ~16GB weights download once and are reused across runs.

# No `set -e`: we capture the container's exit code below and still dump output on failure.
set -uo pipefail

# Runtime params (defaults match RuntimeConfig in run.py). segment_length_s is left unset
# so the model embeds the whole video as one window unless it's exported.
FPS="${FPS:-1}"
MAX_FRAMES="${MAX_FRAMES:-64}"
MAX_LENGTH="${MAX_LENGTH:-8192}"
SEGMENT_LENGTH_S="${SEGMENT_LENGTH_S:-}"   # empty => whole video (model default)

: "${ELV_MODEL_TEST_GPU_TO_USE:=3}"
IMAGE_NAME="${IMAGE_NAME:-qwen3vl-embedding-video-vector}"

cd "$(dirname "$0")"

# Build the --params JSON; include segment_length_s only when set.
PARAMS="{\"fps\": ${FPS}, \"max_frames\": ${MAX_FRAMES}, \"max_length\": ${MAX_LENGTH}"
if [ -n "$SEGMENT_LENGTH_S" ]; then
    PARAMS="${PARAMS}, \"segment_length_s\": ${SEGMENT_LENGTH_S}"
fi
PARAMS="${PARAMS}}"

set -x

rm -rf test-output/
mkdir -p test-output

# Container-side paths (test-files/ is bind-mounted read-only at /elv/test), newline-separated on stdin.
INPUT=$(find test-files/ -maxdepth 1 -type f | sed 's|^test-files/|/elv/test/|' | sort)

echo "$INPUT" | podman run --rm -i \
    --volume="$(pwd)/test-files:/elv/test:ro" \
    --volume="$(pwd)/test-output:/elv/tags:U" \
    --volume=hf_cache:/root/.cache \
    --network host \
    --device "nvidia.com/gpu=${ELV_MODEL_TEST_GPU_TO_USE}" \
    "${IMAGE_NAME}" \
    --output-path /elv/tags/out.jsonl \
    --params "${PARAMS}"

ex=$?

set +x
echo "=== test-output/out.jsonl ==="
find test-output -type f

exit $ex
