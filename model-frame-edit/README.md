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

## Weights

Weights are **not** baked into the image. `Salesforce/blip2-itm-vit-g` (set by
`model.model_id` in `config.yml`) is pulled from the HuggingFace hub the **first time the
model loads**, into the container's HF cache at `HF_HOME=/elv/.hf_cache`. Mount a
persistent volume there — the `hf_cache` podman named volume (see **Build & run**) — so
the download happens **once** and is reused across container runs.

For **reproducibility**, `config.yml` pins the hub snapshot to a specific commit via
`model.revision` (threaded into `from_pretrained` for both the model and the processor).
Set `revision: null` to track the latest commit on the default branch. To use a
local/offline copy, set `model_id` to an absolute path to a mounted weights directory
(`from_pretrained` accepts either; `revision` is ignored for local paths).

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
make build          # or: ./build.sh   (no weights needed at build time)
```

Mount a persistent HF cache at `/elv/.hf_cache` so the weights download once (first run)
and are reused afterwards. Use a podman **named volume** for the cache (podman-managed,
avoids rootless uid-mapping issues) — create it once:

```bash
podman volume create hf_cache
```

The same `hf_cache` volume is shared with `model-qwenvl-edit`; HF keys downloads by repo
id, so the two models coexist in it without collision. Then run:

```bash
podman run --rm \
  --volume=$(pwd)/test-files:/elv/test:ro \
  --volume=$(pwd)/tags:/elv/tags:U \
  --volume=hf_cache:/elv/.hf_cache \
  --network host --device nvidia.com/gpu=3 \
  blip2-frame test/1.mp4
```

- `test/` and `tags/` are **bind mounts** so input frames (`:ro`) and output JSONL
  live directly on the host; only the write-heavy weights cache is a named volume. The
  container reads file paths on stdin and writes tags (`.jsonl`) to `--output-path`, per
  the Eluvio tagging runtime (see `common_ml.tagging.run_helpers.run_default`).
- The **first** run downloads `blip2-itm-vit-g` from the hub into the `hf_cache` volume;
  subsequent runs load straight from that cache.
- Add `--env HF_HUB_OFFLINE=1` to force fully-offline loads once cached, or
  `--env HF_TOKEN=<token>` for gated/rate-limited pulls.
- Swap the cache mount for a bind mount
  (`--volume=$(pwd)/.hf_cache:/elv/.hf_cache`) if a host directory is preferred instead; add `:U` if rootless podman writes it
  with a mapped uid you can't read back.

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
