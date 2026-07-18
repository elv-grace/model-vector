# model-qwenvl-edit

A video-embedding tagger for the Eluvio tagging runtime. It embeds each input
video into a single vector using **Qwen3-VL-Embedding-8B**, and emits it as a
`Vector` tag.

## How it works

The tagger uses Qwen's **native video path**: a whole video (or a time window of
it) is handed to the model at once, so frames are sampled *with temporal position
encoding* and the resulting vector captures motion/ordering.

It plugs into `common_ml` via the `AVModel` interface:

- `embedding/model.py` — `QwenVLVideoEmbedder(AVModel)`: turns one
  `(video, [start_ms, end_ms])` window into one vector via Qwen.
- `run.py` — initializes the model with user parameters for **fps, max_frames, max_length, prompt, normalize, and segment_length_s**, then hands it to `run_default` (the stdin→JSONL tagging daemon).

### Segmentation (scaling to long videos)

A single vector cannot meaningfully represent hours of video, and sampling a
fixed `max_frames` across an 8-hour file yields ~1 frame every several minutes.
So the video can be split into fixed-length segments (configurable by the user with the parameter **segment_length_s**); each segment is embedded over its own time window (dense frame sampling, bounded memory), and the per-segment vectors are output as a list of `Vector` tags.  
The default is to vectorize the entirety of the video without segmenting, yielding a list of one `Vector` tag spanning the whole video duration `[0, duration]`.

## Runtime parameters (`--params` JSON)

| param                 | default | meaning                                                        |
|-----------------------|---------|----------------------------------------------------------------|
| `fps`                 | `1`     | frame sampling rate (Hz) within each embedded window           |
| `max_frames`          | `64`    | max frames Qwen samples per window (bounds memory/compute)     |
| `max_length`          | `8192`  | max token sequence length for the embedder                     |
| `prompt`| `None` | the instruction used by the model to embed the video; default is "Represent the user's input" |
| `normalize`| `None` | whether the output vector(s) should be L2-normalized for cosine similarity; default is to normalize |
| `segment_length_s`    | `None`    | segment duration (s); `null` embeds the whole video as one window |

## Layout

```
model-qwenvl-edit/
├── run.py                       ← entrypoint (builds tagger, runs the daemon)
├── config.py / config.yml       ← weights path + device
├── setup.py                     ← package "embedding" + deps (incl. common-ml from git)
├── embedding/
│   ├── model.py                 ← QwenVLVideoEmbedder (AVModel)
│   ├── qwen3_vl_embedding.py    ← vendored embedder (patched: video_start/video_end trimming)
│   └── Qwen3-VL-Embedding-8B/   ← weights, synced from shared FS at build time (gitignored)
├── Containerfile / build.sh / Makefile
└── tests/test_qwenvl_model.py
```

## Weights

Weights are **not** committed (16GB; gitignored). They live locally under
`embedding/Qwen3-VL-Embedding-8B/` (next to the package code), pointed to by
`storage.embedder_path` in `config.yml`, and the Containerfile bakes that
directory into the image. There is **no shared-source sync step** — the weights
are expected to be present locally before you build (`build.sh` errors out early
if the directory is missing or empty). To use a different location, repoint
`embedder_path`.

Only the model files (safetensors, tokenizer/processor configs) are needed at
runtime. The `scripts/qwen3_vl_embedding.py` bundled inside the HF snapshot is
**not** used — the embedder code is vendored in `embedding/qwen3_vl_embedding.py`
and the model class is instantiated directly (no `trust_remote_code` auto-load).

## Build

Same flow as the other `model-*` taggers (requires podman with the NVIDIA
toolkit, and SSH access to the eluv-io repos with agent forwarding for the
`common-ml` git dependency):

```
ssh-add            # so the container build can fetch common-ml over SSH
./build.sh
```

## Run

```
podman run --rm \
  --volume=$(pwd)/test:/elv/test:ro \
  --volume=$(pwd)/tags:/elv/tags \
  --volume=$(pwd)/.cache:/root/.cache \
  --network host --device nvidia.com/gpu=0 \
  qwenvl-embedding test/1.mp4
```

Output `vector` records are written to the runtime's `--output-path` JSONL.

## Tests

`tests/test_qwenvl_model.py` has fast unit tests (a fake embedder verifies the
`ms`→seconds window conversion and dict wiring — no model load, no GPU) plus an
opt-in end-to-end test.

Run them **in a container** with the production dependency set (the image builds
its own conda env and installs everything from `setup.py`; it does **not** use any
host virtualenv).

```
# inside an image built from setup.py (deps + pytest via the `test` extra)
pip install .[test]
pytest -k "not end_to_end" tests
```

The end-to-end test runs the real model when you point it at weights + a video:

```
QWENVL_EMBEDDER_PATH=/elv/models/Qwen3-VL-Embedding-8B \
QWENVL_TEST_VIDEO=/elv/test/1.mp4 \
pytest -k end_to_end tests      # needs CUDA + weights
```
