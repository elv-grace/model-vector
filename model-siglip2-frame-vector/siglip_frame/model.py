from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from PIL import Image
from transformers import Siglip2ImageProcessor, Siglip2VisionModel

from common_ml.tagging.models.tag_types import FrameTag
from common_ml.tagging.models.frame_based import FrameModel

from siglip_frame.config import RuntimeConfig

# SigLIP 2 NaFlex retrieval checkpoint on the HuggingFace hub: a base ViT image tower and
# a text tower trained together with sigmoid pairwise loss that outputs a pooled 768-d image
# embedding. NaFlex ("native aspect ratio, flexible resolution") means a 16:9 frame
# is not squashed into a square, and the resolution budget is a runtime knob (RuntimeConfig.max_num_patches).
DEFAULT_MODEL_ID = "google/siglip2-base-patch16-naflex"

# Embed the whole frame, so every vector is anchored to the full image in normalized coordinates.
_WHOLE_FRAME_BOX = {"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0}

# Query modalities the checkpoint above can embed INTO this space, stamped on every tag so a
# downstream search UI knows which query kinds an index accepts: text via the checkpoint's
# text tower, image via the same vision tower used here. Declared, not measured -- this
# tagger only ever loads the vision tower -- so pointing `model_id` at a different
# checkpoint means revisiting this list.
QUERY_MODES = ["text", "image"]


class FeatureExtractor(FrameModel):
    """Embeds each video frame (formatted as (H, W, 3) uint8 RGB) into a single search
    vector using SigLIP 2's image tower from HuggingFace transformers.
    The attention-pooling head emits one vector per image, aligned to text."""

    def __init__(
        self,
        cfg: RuntimeConfig,
        model_id: str = DEFAULT_MODEL_ID,
        revision: Optional[str] = None,  # hub commit to pin; None -> default branch. Ignored for local paths.
        dtype: Optional[torch.dtype] = None,  # None -> auto (bf16/fp16 on GPU, fp32 on CPU)
    ) -> None:
        self.config = cfg
        if cfg.max_num_patches < 1:
            # Guard early (before pulling several GB of weights) so a bad param fails fast.
            raise ValueError(f"max_num_patches must be >= 1, got {cfg.max_num_patches!r}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            logger.info("cuda is available, using it")
            if dtype is None:
                # bf16 needs Ampere+ (compute capability >= 8.0); fall back to fp16 otherwise.
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            logger.warning("cuda not available, using cpu (SigLIP 2 will be slow)")
            if dtype is None:
                dtype = torch.float32  # half precision is unstable / slow on CPU
        self.dtype = dtype
        logger.info(f"loading {model_id} (revision={revision}, dtype={self.dtype})")

        # revision pins both the processor and the model to the same hub commit so the
        # whole snapshot is reproducible. Pulled from the hub into HF_HOME on first load
        # and reused from that (mountable) cache afterwards.
        #
        # Vision tower and image processor only because no textual embeddings are made.
        # Loading a sub-model makes transformers log the checkpoint's text-tower keys as
        # UNEXPECTED. That report is the discarded half of the checkpoint, and is expected.
        self.processor = Siglip2ImageProcessor.from_pretrained(model_id, revision=revision)
        self.model = Siglip2VisionModel.from_pretrained(
            model_id, revision=revision, dtype=self.dtype
        ).to(self.device)
        self.model.eval()

        # stamped on every tag; see _embedder_info
        self.model_id = model_id
        self.revision = revision
        # read from the loaded checkpoint rather than hardcoded (768 for -base)
        self.dim = int(self.model.config.hidden_size)

    def tag_frame(self, img: np.ndarray) -> List[FrameTag]:
        vec = self._embed_frame(img)
        # dict(...) so each tag owns its box (the module constant is never shared/mutated).
        return [FrameTag(
            tag="",
            vector=vec.tolist(),
            box=dict(_WHOLE_FRAME_BOX),
            additional_info=self._embedder_info(),
        )]

    def _embedder_info(self) -> Dict:
        """The embedding recipe, stamped on every tag so a query can be embedded into the
        same space after the fact, and so a checkpoint or budget change is visible rather
        than silent.

        Every key here changes the vector, so a query that does not match on all of them
        is not comparable to what is indexed: `embedder`+`revision` identify the exact hub
        snapshot (the text tower of the same checkpoint is what embeds a text query),
        `dim` is the emitted width, `normalize` says whether cosine reduces to a dot
        product, and `max_num_patches` is the NaFlex resolution budget an image query has
        to be preprocessed at.

        `kind` is what the vector IS (this tagger embeds whole frames), so a consumer reads
        it instead of inferring the modality from which positional fields happen to be set.
        `query_modes` is what a query MAY be -- see the module constant."""
        return {
            "embedder": self.model_id,
            "revision": self.revision,
            "dim": self.dim,
            "normalize": self.config.normalize,
            "max_num_patches": self.config.max_num_patches,
            "kind": "frame",
            # list(...) so no tag aliases the module constant
            "query_modes": list(QUERY_MODES),
        }

    def _embed_frame(self, img: np.ndarray) -> np.ndarray:
        inputs = self._preprocess(img)

        with torch.no_grad():
            # .float() before normalizing: dividing in bf16 lands ~0.1% off unit length,
            # which a cosine index reads as a real score difference.
            frame_vector = self.model(**inputs).pooler_output.float()  # (B, 768)
            if self.config.normalize:
                # the pooling head's output is not unit length; normalize so cosine == dot
                frame_vector = F.normalize(frame_vector, p=2, dim=-1)

        # squeeze the batch dim B (one frame in) and return a plain float32 array.
        return frame_vector.squeeze(0).cpu().numpy()

    def _preprocess(self, img: np.ndarray) -> Dict[str, torch.Tensor]:
        """Turn an (H, W, 3) uint8 RGB frame into the NaFlex vision-tower inputs."""
        # Three tensors: the patch sequence, the mask marking which of the `max_num_patches` slots are real 
        # (short sequences are padded up to the budget), and the (height, width) patch grid the position embeddings 
        # are interpolated onto.
        inputs = self.processor(
            images=Image.fromarray(img),
            return_tensors="pt",
            max_num_patches=self.config.max_num_patches,
        )
        # Only pixel_values is float: pixel_attention_mask and spatial_shapes are integer
        # bookkeeping and must keep their own dtypes.
        out = {k: v.to(self.device) for k, v in inputs.items()}
        out["pixel_values"] = out["pixel_values"].to(self.dtype)
        return out
