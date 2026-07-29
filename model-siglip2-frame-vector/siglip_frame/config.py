from dataclasses import dataclass


@dataclass
class RuntimeConfig:
    """Runtime tunables for the SigLIP 2 frame embedder, injected per-request via
    `--params` in run.py."""

    # L2-normalize each emitted frame vector so cosine similarity reduces to a dot
    # product (what the search index expects). Set False to emit the raw features.
    normalize: bool = True

    # NaFlex resolution budget: the frame is resized (aspect ratio preserved) to cover at
    # most this many 16x16 patches, redistributed over the frame's real 16:9 shape.
    # Raising to 576 or 1024 buys detail for small-object queries at quadratic attention cost.
    max_num_patches: int = 256
