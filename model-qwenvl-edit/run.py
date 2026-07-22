from dataclasses import dataclass
from typing import Optional

import setproctitle
from dacite import from_dict

from common_ml.tagging.run_helpers import run_default, catch_errors, get_params

from embedding.model import QwenVLVideoEmbedder
from config import config


@dataclass
class RuntimeConfig:
    # frame sampling rate (Hz) within each embedded window
    fps: float = 1.0
    # max frames Qwen samples per window (bounds memory/compute regardless of length)
    max_frames: int = 64
    # max token sequence length for the embedder
    max_length: int = 8192
    # embedding instruction
    # (should be generic (default is "Represent the user's input") 
    # but can provide some guidance if desired)
    prompt: Optional[str] = None
    # whether the vector should be L2-normalized for cosine similarity comparison
    # None => normalize
    normalize: Optional[bool] = None
    # segment length in seconds; each segment is embedded over its own time window
    # None (or >= duration) => embed the whole video as a single window.
    segment_length_s: Optional[float] = None


if __name__ == "__main__":
    setproctitle.setproctitle("model-qwenvl")

    catch_errors()
    params = get_params()
    params = from_dict(RuntimeConfig, params)

    # one vector per input video
    # or if segment_length_s is provided, then one vector per segment of the input video
    model = QwenVLVideoEmbedder(
        embedder_path=config["model"]["embedder_id"],
        revision=config["model"].get("revision"),
        fps=params.fps,
        max_frames=params.max_frames,
        max_length=params.max_length,
        prompt=params.prompt,
        normalize=params.normalize,
        segment_length_s=params.segment_length_s
    )

    run_default(model)
