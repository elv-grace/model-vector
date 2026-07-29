# model-siglip2-frame-vector — SigLIP 2 frame embedder

Generates one search vector per video frame using [SigLIP 2](https://huggingface.co/docs/transformers/model_doc/siglip2)'s
image tower from HuggingFace `transformers`.

## What it produces

For each sampled frame (from a frame or video), the model emits a single `Tag` with a `vector` field (via `common_ml`'s `FrameModel` or `AVModel.from_frame_model` → `FrameTag` with `vector` path):

- **Checkpoint:** `google/siglip2-base-patch16-naflex` — a base ViT image tower and a
  text tower trained together with the sigmoid pairwise loss, so the pooled **768-d** image
  embedding lives in the space that was contrastively aligned to text. The **same vectors
  serve image→image and text→image search**. Chosen over so400m (1152-d) because the vector
  db caps dimensions at 1024, so the vector can be stored fully without decomposing or a projection step.
- **Pipeline:** the frame is cut into 16×16 patches at its native aspect ratio → the ViT
  encodes them → the attention-pooling head emits one vector → optionally L2-normalized.
- **`box`:** the whole frame (`{x1:0, y1:0, x2:1, y2:1}`) — the model embeds the entire image, not a sub-region.

**NaFlex** ("native aspect ratio, flexible resolution") resizes a 16:9 frame to a *patch budget* rather than squashing
into a square, so nothing is distorted and the resolution is a runtime knob.

### Why SigLIP 2

SigLIP 2 is based on CLIP, which is more suitable for downstream retrieval across frames than BLIP-2. Its pooled output is
a plain dual encoder, so the single vector it emits **is** the trained representation.

It also gives a symmetric text tower (query-side text embedding is one cheap forward pass in
the same space), better fine-grained localization and multilingual/OCR-ish behavior, and
costs less per frame than ViT-g + Q-Former.

**The tradeoff:** there is no late-interaction fallback. SigLIP 2 emits one vector, and its
patch tokens are *not* individually text-aligned (only the pooled head is). Multi-aspect embeddings can be extracted from the individual patch tokens in the last hidden layer and mean pooling and projecting them. Or multi-crop or increase `max_num_patches`.

## Runtime parameters

Injected per request as a JSON `--params` object (see `siglip_frame/config.py`):

| param             | type | default | meaning |
|-------------------|------|---------|---------|
| `normalize`       | bool | `true`  | L2-normalize each emitted frame vector so cosine similarity == dot product (what the search index expects). `false` emits the raw pooled output. |
| `max_num_patches` | int  | `256`   | NaFlex resolution budget: the frame is resized (aspect ratio preserved) to cover at most this many 16×16 patches. `256` is the checkpoint's training budget; `576`/`1024` buy detail for small-object queries at attention cost quadratic in the budget. |

Frame sampling rate (`fps`) and other frame-model plumbing are handled generically by the tagger runtime (`run_default`), not by this config.

## Build

```bash
chmod +x build.sh
make build          # or: ./build.sh   (no weights needed at build time)
```

### Rebuilding after a `common-ml` change (stale layer in podman)

```bash
git submodule update --init --recursive
buildscripts/build_container.bash -t siglip2-frame-vectors:latest . -f Containerfile --no-cache
```

To avoid this issue, consider pinning the dependency to a commit SHA instead of the branch.

## Deployment — requires a persistent `/root/.cache` mount

**This image ships with no weights baked in.** `google/siglip2-base-patch16-naflex` (~1.5 GB) is pulled from the HuggingFace hub the first time the model loads, into the container's HF cache at `HF_HOME=/root/.cache`. This is unlike the baked-weight taggers (`model-shot`, `model-celeb`), which `COPY models` into the image and need no runtime cache.

The deploy/run environment **must mount a
persistent volume at `/root/.cache`** (a named volume or a host bind) mount, or the weights will land in the container's ephemeral writable layer and are lost on `--rm`.
(See `test.sh` for a working invocation (`--volume=hf_cache:/root/.cache`).)

The `hub/` cache is keyed by repo id, so one shared volume can serve this image and `model-qwenvl-video-vector` simultaneously.

## Tests

```bash
pip install -e .[test]
pytest tests/
```

The unit tests stub the model so they run without downloading weights.

Or use `test.sh`.

## Note on the index

The vectors are **768-d**, so the space needs at least
`embedding_size: 768` and anything already indexed has to be re-embedded. 768 is under the
vector db's 1024 ceiling, so **no projection is involved**: the full vector gets stored and compared.

If the so400m (1152-d) model is used (better for fine-grained, local queries (possible multi-aspect-embeddings) and on zero-shot retrieval benchmarks) and has to fit 1024-d, 
**use the uncentered top-k right singular vectors (svd-1024)** (a pure orthonormal rotation, applied identically to frames and queries, then renormalize). 
**Do not truncate, and do not use mean-centered PCA.** 
- SigLIP 2 is not Matryoshka-trained, so truncation is invalid and textbook PCA performs worse than a random projection. The basis is fit on image vectors but must also transform text queries, and subtracting the image mean leaves every query dominated by the modality-gap offset, so they all collapse toward one direction.
Freeze and version the matrix: it must be byte-identical at index time and query time, and a silently refit basis corrupts every comparison with no error.

## Query side must use the same checkpoint

Text→frame and image→frame search only work if the query embedder is this exact checkpoint:

- Use `Siglip2Processor` / `AutoProcessor` for the text side. SigLIP was trained with the
  tokenizer padded to a fixed 64 tokens; the processor defaults to
  `padding="max_length", max_length=64` for you, but a bare tokenizer call does not, and the
  mismatch degrades scores quietly rather than erroring.
- In `transformers` 5, `get_text_features()` returns the model output object, not the pooled tensor (4.x returned the tensor). Take `.pooler_output`.
- The query side needs the **text tower** (unused and not loaded here), so it should load the same checkpoint separately.
