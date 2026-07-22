# model-frame-edit — BLIP-2 frame embedder

Generates one search vector per video frame using BLIP-2's image-text-retrieval stack
(ViT-g + Q-Former + vision projection, **no LLM**) from HuggingFace `transformers`.

Using BLIP-2 from [HuggingFace](https://huggingface.co/docs/transformers/model_doc/blip-2) `transformers` (rather than Salesforce's
[`LAVIS`](https://github.com/salesforce/LAVIS)) for better dependency management, GPU
optimization, code maintenance, and serving compatibility.

## What it produces

For each sampled frame (from a frame or video), the model emits a single `Vector` (via `common_ml`'s
`FrameModel` or `AVModel.from_frame_model` → `FrameVector` path):

- **Checkpoint:** `Salesforce/blip2-itm-vit-g` — the *retrieval* checkpoint. Its image
  embeddings live in the 256-d space that was contrastively aligned to text, so the
  **same vectors serve both image→image and text→image search**.
- **Pipeline:** ViT-g encodes the frame → the Q-Former distills it into 32 query tokens
  → the vision projection maps each token to the 256-d contrastive space and
  L2-normalizes it → the 32 tokens are **mean-pooled** into one vector per frame.
- **`box`:** the whole frame (`{x1:0, y1:0, x2:1, y2:1}`) — the model embeds the entire
  image, not a sub-region.

## Runtime parameters

Injected per request as a JSON `--params` object (see `blip_frame/config.py`):

| param       | type   | default  | meaning |
|-------------|--------|----------|---------|
| `normalize` | bool   | `true`   | L2-normalize each emitted frame vector so cosine similarity == dot product (what the search index expects). `false` emits the raw pooled mean. |
| `pooling`   | string | `"mean"` | How to reduce the Q-Former's 32 aligned tokens. Only `"mean"` (one vector/frame) is implemented; `"tokens"` is reserved for the 32-token / max-sim extension below. |

Frame sampling rate (`fps`) and other frame-model plumbing are handled generically by
the tagger runtime (`run_default`), not by this config.

## Build

```bash
chmod +x build.sh
make build          # or: ./build.sh   (no weights needed at build time)
```

## Deployment — requires a persistent `/root/.cache` mount

**This image ships with no weights baked in.** `Salesforce/blip2-itm-vit-g` is pulled from
the HuggingFace hub the first time the model loads, into the container's HF cache at
`HF_HOME=/root/.cache`. This is unlike the baked-weight taggers (`model-shot`,
`model-celeb`), which `COPY models` into the image and need no runtime cache.

**Consequence for whoever runs the container:** the deploy/run environment **must mount a
persistent volume at `/root/.cache`** (a named volume or a host bind mount). Without it,
the weights land in the container's ephemeral writable layer and are lost on `--rm`,
forcing a full re-download on **every** container start.

- The buildscripts only **build and push** the image (`make deploy` = tag + push to the
  registry); they do **not** run it or configure any mount. The tagging runtime that
  launches the container owns the cache mount.
- A **named volume** (or host bind mount) at `/root/.cache` survives `--rm`; the writable
  layer does not. See `test.sh` for a working invocation (`--volume=hf_cache:/root/.cache`).
- The `hub/` cache is keyed by repo id, so one shared volume can serve this image and
  `model-qwenvl-video-vector` simultaneously.
- Tag output (`--output-path`) is unaffected — that's plumbed by the runtime as usual;
  only the weights cache needs this mount.

## Tests

```bash
pip install -e .[test]
pytest tests/
```

The unit tests stub the model so they run without downloading weights.

Or use `test.sh`.

## Future work: all 32 tokens (search retrieval)

BLIP-2 natively scores retrieval by comparing **all 32 image tokens** to the text token
via max-similarity (ColBERT-style late interaction), not by mean pooling. Setting `pooling="tokens"` will emit the 32 projected, normalized tokens per frame and do
max-sim at query time. (`_project_tokens()` already returns the full `(32, 256)` matrix,
so need to teach the index to max-pool over the 32 `FrameVector`s per frame.)
