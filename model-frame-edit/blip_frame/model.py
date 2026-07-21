from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from PIL import Image
from transformers import Blip2Processor, Blip2ForImageTextRetrieval

from common_ml.tagging.models.tag_types import FrameVector
from common_ml.tagging.models.frame_based import FrameModel

from blip_frame.config import RuntimeConfig

# BLIP-2 retrieval checkpoint on the HuggingFace hub: ViT-g + Q-Former + image/text
# projection heads (no LLM) (256-d image embeddings for image->image and text->image search).
DEFAULT_MODEL_ID = "Salesforce/blip2-itm-vit-g"

# Embed the whole frame, so every vector is anchored to the full image in normalized coordinates.
_WHOLE_FRAME_BOX = {"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0}


class FeatureExtractor(FrameModel):
    """Embeds each video frame (formatted as (H, W, 3) uint8 RGB) into a single search
    vector using BLIP-2's image-text-retrieval stack (ViT-g + Q-Former + vision
    projection, no LLM) from HuggingFace transformers.

    The Q-Former produces 32 query tokens per frame; the vision projection maps each to
    the 256-d contrastive space and L2-normalizes it. With `pooling="mean"` the 32
    aligned tokens are averaged into one vector per frame."""

    def __init__(
        self,
        cfg: RuntimeConfig,
        model_id: str = DEFAULT_MODEL_ID,
        dtype: Optional[torch.dtype] = None,  # None -> auto (bf16/fp16 on GPU, fp32 on CPU)
    ) -> None:
        self.config = cfg
        if cfg.pooling != "mean":
            # Guard early (before pulling several GB of weights) so a bad param fails fast.
            raise NotImplementedError(
                f"pooling={cfg.pooling!r} is not implemented; only 'mean' (one vector per frame) is supported"
            )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            logger.info("cuda is available, using it")
            if dtype is None:
                # bf16 needs Ampere+ (compute capability >= 8.0); fall back to fp16 otherwise.
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            logger.warning("cuda not available, using cpu (BLIP-2 will be slow)")
            if dtype is None:
                dtype = torch.float32  # half precision is unstable / slow on CPU
        self.dtype = dtype
        logger.info(f"loading {model_id} (dtype={self.dtype})")

        self.processor = Blip2Processor.from_pretrained(model_id)
        self.model = Blip2ForImageTextRetrieval.from_pretrained(model_id, dtype=self.dtype).to(self.device)
        self.model.eval()

    def tag_frame(self, img: np.ndarray) -> List[FrameVector]:
        vec = self._embed_frame(img)
        # dict(...) so each tag owns its box (the module constant is never shared/mutated).
        return [FrameVector(vector=vec.tolist(), box=dict(_WHOLE_FRAME_BOX))]

    def _embed_frame(self, img: np.ndarray) -> np.ndarray:
        # Processor normalizes an (H, W, 3) uint8 RGB image into pixel_values; cast them
        # to the model's dtype so the forward pass runs entirely in that precision.
        pixel_values = self.processor(images=Image.fromarray(img), return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device, self.dtype)

        with torch.no_grad():
            tokens = self._project_tokens(pixel_values)  # (B, 32, 256), each token L2-normalized
            frame_vector = tokens.mean(dim=1)             # mean pool the 32 tokens -> (B, 256)
            if self.config.normalize:
                # mean of unit vectors isn't unit length; renormalize so cosine == dot.
                frame_vector = F.normalize(frame_vector, p=2, dim=-1)

        # squeeze the batch dim B (one frame in) and return a plain float32 array.
        return frame_vector.squeeze(0).float().cpu().numpy()

    def _project_tokens(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run ViT -> Q-Former -> vision projection to get the 32 aligned, L2-normalized
        query-token embeddings. (Mirrors the image branch of the model's own Image-Text
        Contrastive forward pass where matched image/text are pushed to have high similarity
        and 'close' means 'semantically matching' in the contrastive space, but without 
        needing a paired text (caption) input.)

        Returns a (B, 32, 256) tensor (B = batch size = 1 for the per-frame path)."""
        m = self.model
        image_embeds = m.vision_model(pixel_values=pixel_values, return_dict=True).last_hidden_state # sequence of patch embeddings from the ViT image encoder
        image_atts = torch.ones(image_embeds.shape[:-1], dtype=torch.long, device=image_embeds.device) # all-1s attention mask the Q-Former uses to cross-attend all patches

        query_tokens = m.query_tokens.expand(image_embeds.shape[0], -1, -1) # learned parameter of 32 "queries" (tokens) corresponding to image features
        query_output = m.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        ).last_hidden_state  # (B, 32, 768), B == 1 # 32 summary vectors

        # project to the contrastive space and L2-normalize per token (intrinsic to
        # BLIP-2's aligned embedding definition).
        query_output = query_output.to(m.vision_projection.weight.dtype)
        return F.normalize(m.vision_projection(query_output), dim=-1)  # trained linear layer => (B, 32, 256)
