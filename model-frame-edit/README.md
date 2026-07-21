# model-frame-edit — BLIP-2 frame embedder

Generates one search vector per video frame using BLIP-2's image-text-retrieval stack
(ViT-g + Q-Former + vision projection, **no LLM**) from HuggingFace `transformers`.

Using BLIP-2 from HuggingFace `transformers` (rather than Salesforce's
[`LAVIS`](https://github.com/salesforce/LAVIS)) for better dependency management, GPU
optimization, code maintenance, and serving compatibility.

## What it produces

For each sampled frame the model emits a single `Vector` (via `common_ml`'s
`FrameModel` → `FrameVector` path):

- **Checkpoint:** `Salesforce/blip2-itm-vit-g` — the *retrieval* checkpoint. Its image
  embeddings live in the 256-d space that was contrastively aligned to text, so the
  **same vectors serve both image→image and text→image search**.
- **Pipeline:** ViT-g encodes the frame → the Q-Former distills it into 32 query tokens
  → the vision projection maps each token to the 256-d contrastive space and
  L2-normalizes it → the 32 tokens are **mean-pooled** into one vector per frame.
- **`box`:** the whole frame (`{x1:0, y1:0, x2:1, y2:1}`) — the model embeds the entire
  image, not a sub-region.

Weights are pulled from the HuggingFace hub at load time; the container bakes them into
its HF cache at build time so no network is hit per request.

## Runtime parameters

Injected per request as a JSON `--params` object (see `blip_frame/config.py`):

| param       | type   | default  | meaning |
|-------------|--------|----------|---------|
| `normalize` | bool   | `true`   | L2-normalize each emitted frame vector so cosine similarity == dot product (what the search index expects). `false` emits the raw pooled mean. |
| `pooling`   | string | `"mean"` | How to reduce the Q-Former's 32 aligned tokens. Only `"mean"` (one vector/frame) is implemented; `"tokens"` is reserved for the 32-token / max-sim extension below. |

Frame sampling rate (`fps`) and other frame-model plumbing are handled generically by
the tagger runtime (`run_default`), not by this config.

## Build & run

```bash
make build          # or: ./build.sh   (bakes the weights into the image)
```

The container reads file paths on stdin and writes tags (`.jsonl`) to `--output-path`,
per the Eluvio tagging runtime (see `common_ml.tagging.run_helpers.run_default`).

## Tests

```bash
pip install -e .[test]
pytest tests/
```

The unit tests stub the model so they run without downloading weights; an end-to-end
test on real weights can be added later (see `model-qwenvl-edit` for that pattern).

## Future work: all 32 tokens (late-interaction retrieval)

BLIP-2 natively scores retrieval by comparing **all 32 image tokens** to the text token
via max-similarity (ColBERT-style late interaction), not by mean pooling. The natural
upgrade is `pooling="tokens"`: emit the 32 projected, normalized tokens per frame and do
max-sim at query time. `_project_tokens()` already returns the full `(32, 256)` matrix,
so this is a matter of emitting 32 `FrameVector`s per frame and teaching the index to
max-pool over them.
