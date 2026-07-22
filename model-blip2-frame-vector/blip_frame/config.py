from dataclasses import dataclass


@dataclass
class RuntimeConfig:
    """Runtime tunables for the BLIP-2 frame embedder, injected per-request via
    `--params` in run.py."""

    # L2-normalize each emitted frame vector so cosine similarity reduces to a dot
    # product (what the search index expects). Set False to emit the raw features.
    normalize: bool = True

    # How to reduce the Q-Former's (32, hidden) query-token matrix into per-frame output:
    #   "mean"   -> one vector per frame (mean over the 32 query tokens)
    #   "tokens" -> emit all 32 token vectors per frame for max-similarity retrieval [not yet implemented]
    pooling: str = "mean"
