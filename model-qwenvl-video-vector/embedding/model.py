
from typing import Dict, List, Optional

import torch
from loguru import logger
import numpy as np

from common_ml.tagging.models.av import AVModel
from common_ml.video_processing import get_duration
from common_ml.tagging.messages import Tag

from embedding.qwen3_vl_embedding import Qwen3VLEmbedder, EMBEDDING_DIM

# Query modalities Qwen3-VL-Embedding can embed INTO this space, stamped on every tag so a
# downstream search UI knows which query kinds an index accepts. All three, because it is a
# unified multimodal embedder: `format_model_input` takes text, image and video and pools
# every one of them off the same text tower. Declared, not measured, so pointing
# `embedder_id` at a different checkpoint means revisiting this list.
QUERY_MODES = ["text", "image", "video"]


class QwenVLVideoEmbedder(AVModel):
    """Embeds a video into one vector per segment using Qwen3-VL-Embedding's
    native video path, and is itself an `AVModel`.
    Qwen samples frames and returns one temporally-aware vector for each segment.
    `tag()` splits the input video into user-defined `segment_length_s` segments.
    When `segment_length_s` is not passed as a user parameter, the segment is the whole video."""

    def __init__(
        self,
        embedder_path: str,
        revision: Optional[str] = None,  # hub commit to pin; None -> default branch. Ignored for local paths.
        fps: float = 1.0,
        max_frames: int = 64,
        max_length: int = 8192,
        # Output vector width, MRL-truncated from the checkpoint's native 4096. Fixed by
        # config.yml (deployment-level), NOT a per-request param: every vector in a search
        # index must share one width, so it cannot vary request to request.
        embedding_dim: int = EMBEDDING_DIM,
        prompt: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,  # None -> bfloat16 if the GPU supports it, else float16
        normalize: Optional[bool] = None,
        segment_length_s: Optional[float] = None
    ):
        if torch.cuda.is_available():
            logger.info("cuda is available, using it")
        else:
            logger.warning("cuda not available, using cpu (Qwen3-VL-8B will be very slow)")

        if dtype is None:
            # bfloat16 needs Ampere+ (compute capability >= 8.0); torch reports this
            # via is_bf16_supported(). Fall back to float16 on older GPUs / CPU.
            dtype = (
                torch.bfloat16
                if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                else torch.float16
            )
            logger.info(f"dtype not specified, using {dtype} based on hardware support")

        self.embedder = Qwen3VLEmbedder(
            model_name_or_path=embedder_path,
            revision=revision,
            dtype=dtype,
            fps=fps,
            max_frames=max_frames,
            max_length=max_length,
            embedding_dim=embedding_dim,
        )
        self.embedding_dim = embedding_dim
        self.fps = fps
        self.max_frames = max_frames
        self.prompt = prompt
        self.normalize = normalize
        self.segment_length_s = segment_length_s
        # stamped on every tag; see _embedder_info
        self.model_id = embedder_path
        self.revision = revision

    @staticmethod
    def _ms_to_seconds(ms: Optional[int]) -> Optional[float]:
        return None if ms is None else ms / 1000.0

    def _embedder_info(self) -> Dict:
        """The embedding recipe, stamped on every tag so a query can be embedded into the
        same space after the fact, and so a checkpoint or width change is visible rather
        than silent.

        Every key here changes the vector, so a query that does not match on all of them
        is not comparable to what is indexed: `embedder`+`revision` identify the exact hub
        snapshot, `dim` is the MRL width the vector was truncated and re-normalized to,
        `normalize` says whether cosine reduces to a dot product, `prompt` is the
        instruction the input was embedded under (an embedding model conditions on it, so
        a query embedded under a different instruction lands elsewhere), and
        `fps`/`max_frames`/`max_length` are the sampling budget a *video* query has to
        match (a text query only shares `max_length`).

        The sampling values are read off the embedder rather than off `self`, because that
        is where the token-budget clamp (`_clamp_max_frames`) has been applied -- these are
        the values the vector was actually produced with.

        `query_modes` is what a query MAY be -- see the module constant."""
        return {
            "embedder": self.model_id,
            "revision": self.revision,
            "dim": self.embedding_dim,
            # the resolved flag, not the None sentinel (_embed_video does the same)
            "normalize": True if self.normalize is None else self.normalize,
            "prompt": self.prompt or self.embedder.default_instruction,
            "fps": self.fps,
            "max_frames": self.embedder.max_frames,
            "max_length": self.embedder.max_length,
            # list(...) so no tag aliases the module constant
            "query_modes": list(QUERY_MODES),
        }

    def tag(self, fpath: str) -> List[Tag]:
        logger.debug(f"Qwen3-VL embedding {fpath}")

        duration_s = get_duration(fpath)
        duration_ms = round(duration_s * 1000)

        # Build segment [start_ms, end_ms] windows over the media.
        if self.segment_length_s is None or self.segment_length_s <= 0 or duration_s <= self.segment_length_s:
            windows = [(0, duration_ms)]
        else:
            seg_ms = round(self.segment_length_s * 1000)
            # A trailing remainder shorter than one frame period maps to an empty frame
            # range (start frame >= last frame, because the container's reported duration
            # runs slightly past the last decodable frame), which makes qwen_vl_utils
            # raise "Invalid time range". Include tail in the previous window instead.
            min_window_ms = round(1000.0 / self.fps) if self.fps > 0 else 0
            windows = []
            start_ms = 0
            while start_ms < duration_ms:
                end_ms = min(start_ms + seg_ms, duration_ms)
                if windows and (end_ms - start_ms) < min_window_ms:
                    windows[-1] = (windows[-1][0], end_ms)
                else:
                    windows.append((start_ms, end_ms))
                start_ms += seg_ms

        out: List[Tag] = []
        segment_vecs: List[np.ndarray] = []
        single_window = len(windows)==1
        # invariant across every segment of this media, so built once
        embedder_info = self._embedder_info()
        for (start_ms, end_ms) in windows:
            # Guard each segment: one bad window must not abort the whole
            # media (which would discard every other segment's work and
            # emit only an Error). Skip it and keep going.
            try:
                vec = np.asarray(
                    self._embed_video(fpath, start_ms, end_ms),
                    dtype=np.float64,
                )
            except Exception as e:
                logger.warning(
                    f"skipping segment [{start_ms}, {end_ms}]ms of {fpath}: {e!r}"
                )
                continue
            if vec.size == 0:
                logger.warning(
                    f"empty embedding for segment [{start_ms}, {end_ms}]ms of {fpath}; skipping"
                )
                continue

            # _embed_video already applied `self.normalize`, so segment vectors are
            # normalized (or raw) as requested -- emit them as-is.
            segment_vecs.append(vec)
            # if no segmenting (just one vector) -> start 0 and end time duration_ms, else -> actual timestamps
            st, et = (0, duration_ms) if single_window else (start_ms, end_ms)
            out.append(Tag(
                tag="",
                vector=vec.tolist(),
                start_time=st,
                end_time=et,
                source_media=fpath,
                track="",
                frame_info=None,
                # dict(...) so each tag owns its copy and no two can alias
                additional_info=dict(embedder_info),
            ))

        if not segment_vecs:  # no usable segments -> emit no vector
            logger.warning(f"no usable segments for {fpath}; emitting no vector")
            return out # empty list []

        # no pooling segment-vectors into one vector
        # if a long video was processed by segment, then return all the segment vectors
        return out

    def _embed_video(self, fpath: str, 
                     start_ms: Optional[int] = None, end_ms: Optional[int] = None) -> List[float]:
        item = {"video": fpath}
        if self.prompt:
            item["instruction"] = self.prompt
        # Qwen's video reader expects the window in seconds; None -> media boundary.
        start_s = self._ms_to_seconds(start_ms)
        end_s = self._ms_to_seconds(end_ms)
        if start_s is not None:
            item["video_start"] = start_s
        if end_s is not None:
            item["video_end"] = end_s

        # None -> this embedder's default (L2-normalize)
        do_normalize = True if self.normalize is None else self.normalize
        # process() returns embeddings of shape (batch, dim); one input
        embeddings = self.embedder.process([item], normalize=do_normalize)
        return embeddings[0].float().cpu().numpy().tolist()
