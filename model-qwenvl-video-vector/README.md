# model-qwenvl-video-vector - Qwen3-VL-Embedding video embedder

A video-embedding tagger for the Eluvio tagging runtime. It embeds each input
video into a single vector using **Qwen3-VL-Embedding-8B**, and emits it as a
`Tag` with `vector` field.

## How it works

The tagger uses Qwen's **native video path**: a whole video (or a time window of
it) is handed to the model at once, so frames are sampled *with temporal position
encoding* and the resulting vector captures motion/ordering.

### Output dimension (1024, MRL-truncated)

The checkpoint pools a **4096-d** vector (the text tower's `hidden_size`), but the
tagger emits **1024-d** vectors. `Qwen3-VL-Embedding` is trained with **Matryoshka
Representation Learning** (model card: *MRL Support: Yes*, user-defined widths
**64–4096**), which front-loads information into the leading coordinates — so the
shorter vector is the first 1024 dims, **re-normalized at that width**. Order matters:
truncating an already-normalized 4096-d vector would leave a norm < 1 and distort
cosine scores, so `process()` truncates *then* normalizes.

The width is set by `embedding_dim` in `config.yml`, alongside `embedder_id`/`revision` —
it is **deployment config, not a `--params` runtime tunable**, because every vector in a
search index must share one width. Changing it invalidates already-indexed vectors;
they have to be re-embedded.

It plugs into `common_ml` via the `AVModel` interface:

- `embedding/model.py` — `QwenVLVideoEmbedder(AVModel)`: turns one
  `(video, [start_ms, end_ms])` window into one vector via Qwen.
- `run.py` — initializes the model with user parameters for **fps, max_frames, max_length, prompt, normalize, and segment_length_s**, then hands it to `run_default` (the stdin→JSONL tagging daemon).

### Segmentation (scaling to long videos)

A single vector cannot meaningfully represent hours of video, and sampling a
fixed `max_frames` across an 8-hour file yields ~1 frame every several minutes.
So the video can be split into fixed-length segments (configurable by the user with the parameter **segment_length_s**); each segment is embedded over its own time window (dense frame sampling, bounded memory), and the per-segment vectors are output as a list of `Tag` with `vector` field tags.  
The default is to vectorize the entirety of the video without segmenting, yielding a list of one `Tag` with `vector` tag spanning the whole video duration.

## Runtime parameters (`--params` JSON)

| param                 | default | meaning                                                        |
|-----------------------|---------|----------------------------------------------------------------|
| `fps`                 | `1`     | frame sampling rate (Hz) within each embedded window           |
| `max_frames`          | `64`    | max frames Qwen samples per window (bounds memory/compute)     |
| `max_length`          | `8192`  | max token sequence length for the embedder                     |
| `prompt`| `None` | the instruction used by the model to embed the video; default is "Represent the user's input" |
| `normalize`| `None` | whether the output vector(s) should be L2-normalized for cosine similarity; default is to normalize |
| `segment_length_s`    | `None`    | segment duration (s); `null` embeds the whole video as one window |

## Vector metadata (`additional_info`)

Every emitted tag carries the embedding recipe, so a **downstream query can be embedded
into the same space** without the index having to record which model built it (it doesn't —
`GET /indexes/{qid}` returns only `{qid, vector_size}`). Same `embedder`/`dim` convention as
the sibling vector taggers (`model-detection`, `model-celeb-vector`).

| key | example | why a query has to match it |
|-----|---------|------------------------------|
| `embedder` | `Qwen/Qwen3-VL-Embedding-8B` | the checkpoint |
| `revision` | `2c456551…` | the exact hub commit, so "the same model" really is the same weights |
| `dim` | `1024` | the MRL width the vector was truncated **and re-normalized** to |
| `normalize` | `true` | whether cosine reduces to a dot product. Resolved, never the `None` sentinel |
| `prompt` | `Represent the user's input.` | the instruction the input was embedded under. An embedding model *conditions* on it, so a query embedded under a different one lands elsewhere |
| `fps` | `1.0` | frame sampling rate — part of the budget a **video** query has to match |
| `max_frames` | `64` | the **effective** cap, after the embedder's token-budget clamp (`_clamp_max_frames`), i.e. what the vector was really produced with |
| `max_length` | `8192` | token sequence length; the one sampling key a **text** query also shares |
| `kind` | `video` | what the vector *is*. It cannot be inferred from the timestamps: an unsegmented whole-video tag carries `start_time == end_time == 0` |
| `query_modes` | `["text", "image", "video"]` | which query kinds this space accepts — all three, since `format_model_input` pools text, image and video off the same text tower. Declared for the checkpoint above, so an `embedder_id` swap means revisiting `QUERY_MODES` in `embedding/model.py` |

`query_modes` gates *which* queries to offer, not *how* to embed one — the query-side code
still keys off `embedder`, because the instruction + chat-template path is model-specific.

**Where it travels.** The vectorstore stores `additional_info` as JSONB and returns it
**verbatim on every search hit** (`VectorResponse.additional_info`), so a consumer gets the
recipe on the same row as the vector — no second lookup against the content object. It is
opaque to the index, though: *not indexed and not filterable*, so it cannot narrow a search.
The filterable fields stay `qids`, `sources`, `track` and the start/end time ranges.

## Build

Same flow as the other `model-*` taggers (requires podman with the NVIDIA
toolkit, and SSH access to the eluv-io repos with agent forwarding for the
`common-ml` git dependency):

```
ssh-add            # so the container build can fetch common-ml over SSH
chmod +x build.sh
./build.sh         # no weights required at build time; they download at first run
```

## Deployment — requires a persistent `/root/.cache` mount

**This image ships with no weights baked in.** `Qwen/Qwen3-VL-Embedding-8B` (~16GB) is
pulled from the HuggingFace hub the first time the model loads, into the container's HF
cache at `HF_HOME=/root/.cache`. This is unlike the baked-weight taggers (`model-shot`,
`model-celeb`), which `COPY models` into the image and need no runtime cache.

**Consequence for whoever runs the container:** the deploy/run environment **must mount a
persistent volume at `/root/.cache`** (a named volume or a host bind mount). Without it,
the weights land in the container's ephemeral writable layer and are lost on `--rm`,
forcing a full ~16GB re-download on **every** container start.

## Tests

`tests/test_qwenvl_model.py` has fast unit tests (a fake embedder verifies the
`ms`→seconds window conversion and dict wiring — no model load, no GPU) plus an
opt-in end-to-end test.

Run them **in a container** with the production dependency set (the image builds
its own conda env and installs everything from `setup.py`; it does **not** use any
host virtualenv). Running `pytest` against a host interpreter may resolve a different/older `common-ml` than the `vector-tags` snapshot pinned in `setup.py` (whose `Tag` carries the `vector` field), so the suite can fail with `Tag ... unexpected keyword 'vector'` for environment reasons unrelated to the code.

To build a **test-capable image** independent of the host interpreter, add `pytest` via the `test` extra (the default build omits it), then run the suite. The image's `ENTRYPOINT` is `run.py`, so override it to invoke pytest. Use **`python -m pytest`** (not the bare `pytest` executable): `pip install .` puts an installed `embedding` package on `sys.path`, and `-m` runs with the working dir (`/elv`) first so the *source* tree shadows it — the same import order `run.py` relies on.

```
# build once with the test extra
podman build --build-arg INSTALL_TEST=true -t qwen3vl-embedding-video-vector:test -f Containerfile .

# fast unit tests (no model load, no GPU)
podman run --rm --entrypoint /opt/conda/envs/mlpod/bin/python \
    qwen3vl-embedding-video-vector:test -m pytest -k "not end_to_end" tests
```

The end-to-end test runs the real model when you point it at a model + a video.
`QWENVL_EMBEDDER_PATH` accepts either a hub id or a local weights path:

```
podman run --rm --entrypoint /opt/conda/envs/mlpod/bin/python \
    --volume=hf_cache:/root/.cache \
    --volume="$(pwd)/test-files:/elv/test:ro" \
    --device nvidia.com/gpu=0 \
    --env QWENVL_EMBEDDER_PATH=Qwen/Qwen3-VL-Embedding-8B \
    --env QWENVL_TEST_VIDEO=/elv/test/USfootball10s.mp4 \
    qwen3vl-embedding-video-vector:test -m pytest -k end_to_end tests   # needs CUDA; downloads weights if not cached
```

Or use `test.sh` for a container smoke test of the entrypoint.