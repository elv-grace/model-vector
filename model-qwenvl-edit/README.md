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
│   └── qwen3_vl_embedding.py    ← vendored embedder (patched: video_start/video_end trimming)
├── Containerfile / build.sh / Makefile
└── tests/test_qwenvl_model.py
```

## Weights

Weights are **not** baked into the image and **not** committed. The model
(`Qwen/Qwen3-VL-Embedding-8B`, set by `model.embedder_id` in `config.yml`) is pulled
from the HuggingFace hub the **first time the model loads**, into the container's HF
cache at `HF_HOME=/elv/.hf_cache`.

Mount a persistent host volume there (see **Run**) so the ~16GB download happens **once**
and is reused across container runs instead of being re-downloaded on every start. The
build no longer needs the weights present, so `build.sh` has no weight-presence check.

For **reproducibility**, `config.yml` pins the hub snapshot to a specific commit via
`model.revision` (threaded into `from_pretrained` for both the model and the processor),
so a runtime pull always fetches the exact same weights/config/tokenizer even if the hub
repo is updated later. Set `revision: null` to track the latest commit on `main`.

To use a local/offline copy instead of the hub, set `embedder_id` to an **absolute path**
to a mounted weights directory — `from_pretrained` accepts either a hub id or a path, and
`config.py` leaves it untouched (it is not under the path-resolved `storage` key).

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
./build.sh         # no weights required at build time; they download at first run
```

## Run

Mount a persistent HF cache at `/elv/.hf_cache` so the weights download once (first run)
and are reused afterwards:

```
podman run --rm \
  --volume=$(pwd)/test:/elv/test:ro \
  --volume=$(pwd)/tags:/elv/tags \
  --volume=$(pwd)/.hf_cache:/elv/.hf_cache \
  --network host --device nvidia.com/gpu=0 \
  qwenvl-embedding test/1.mp4
```

- The **first** run downloads `Qwen/Qwen3-VL-Embedding-8B` from the hub into
  `.hf_cache/` on the host (~16GB); subsequent runs load straight from that cache.
- For a **gated** or rate-limited download, pass a token: `--env HF_TOKEN=<token>`.
- To run fully offline once cached, add `--env HF_HUB_OFFLINE=1`.

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

The end-to-end test runs the real model when you point it at a model + a video.
`QWENVL_EMBEDDER_PATH` accepts either a hub id or a local weights path:

```
QWENVL_EMBEDDER_PATH=Qwen/Qwen3-VL-Embedding-8B \
QWENVL_TEST_VIDEO=/elv/test/1.mp4 \
pytest -k end_to_end tests      # needs CUDA; downloads weights if not cached
```
