# ---------------------------------------------------------------------------
# The embedder for the tagger, imported by `embedding/model.py`
#  (`from embedding.qwen3_vl_embedding import Qwen3VLEmbedder`).
#
#   format_model_input() and process() accept `video_start` / `video_end`
#   (seconds) and pass them into the video content element, so qwen_vl_utils
#   trims decoding/sampling to that time window. This powers the dense
#   per-segment embedding by the QwenVLVideoEmbedder tagger model.
# ---------------------------------------------------------------------------
import os
import torch
import torch.nn.functional as F
import unicodedata
import numpy as np
import logging

from PIL import Image
from urllib.parse import urlparse
from dataclasses import dataclass
from typing import Optional, List, Union, Dict, Any
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLPreTrainedModel, Qwen3VLModel, Qwen3VLConfig
from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
from transformers.modeling_outputs import ModelOutput
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs
from transformers.cache_utils import Cache
from transformers.utils.generic import check_model_inputs
from qwen_vl_utils.vision_process import process_vision_info

logger = logging.getLogger(__name__)

# Constants for configuration
MAX_LENGTH = 8192
IMAGE_BASE_FACTOR = 16
IMAGE_FACTOR = IMAGE_BASE_FACTOR * 2
MIN_PIXELS = 4 * IMAGE_FACTOR * IMAGE_FACTOR
MAX_PIXELS = 1800 * IMAGE_FACTOR * IMAGE_FACTOR
FPS = 1
MAX_FRAMES = 64
FRAME_MAX_PIXELS = 768 * IMAGE_FACTOR * IMAGE_FACTOR
MAX_TOTAL_PIXELS = 10 * FRAME_MAX_PIXELS
PAD_TOKEN = "<|endoftext|>"

# --- video token-budget constants ------------------------------------------
# Qwen3-VL turns a video into placeholder tokens; the count is driven by how many
# frames x pixels qwen_vl_utils samples. If that count exceeds `max_length`, the
# processor's `truncation=True` chops `input_ids`, the video-token count in the ids
# no longer matches the count in the chat template, and the processor raises
# "Mismatch in `video` token count ...". To guarantee a (possibly degraded) vector
# instead of an error, we size the video to fit `max_length` up front and keep a
# shrinking retry schedule as a safety net.
#
# One merged video token covers IMAGE_FACTOR x IMAGE_FACTOR pixels (patch 16 x
# spatial_merge 2), and TEMPORAL_PATCH_SIZE frames are merged in time, so
#     video_tokens ~= total_sampled_pixels / (IMAGE_FACTOR**2 * TEMPORAL_PATCH_SIZE)
# which we invert to turn a token budget into a `total_pixels` budget.
TEMPORAL_PATCH_SIZE = 2   # frames merged per temporal patch (config: temporal_patch_size)
VIDEO_MIN_TOKEN_NUM = 128  # qwen_vl_utils floor: min spatial tokens per frame
TEXT_TOKEN_RESERVE = 256  # tokens left for system prompt / instruction / template
PIXEL_SAFETY = 0.9        # keep the estimate under the true count (matches qwen_vl_utils' 0.9)
COARSE_MAX_FRAMES = 8     # last-resort: a handful of frames at ~min resolution

# Define output structure for embeddings
@dataclass
class Qwen3VLForEmbeddingOutput(ModelOutput):
    last_hidden_state: Optional[torch.FloatTensor] = None
    attention_mask: Optional[torch.Tensor] = None

# Define model class to compute embeddings
class Qwen3VLForEmbedding(Qwen3VLPreTrainedModel):
    _checkpoint_conversion_mapping = {}
    accepts_loss_kwargs = False
    config: Qwen3VLConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3VLModel(config)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        self.model.set_decoder(decoder)

    def get_decoder(self):
        return self.model.get_decoder()

    # Extract video features from model
    def get_video_features(self, pixel_values_videos: torch.FloatTensor,
                           video_grid_thw: Optional[torch.LongTensor] = None):
        return self.model.get_video_features(pixel_values_videos, video_grid_thw)

    # Extract image features from model
    def get_image_features(self, pixel_values: torch.FloatTensor,
                           image_grid_thw: Optional[torch.LongTensor] = None):
        return self.model.get_image_features(pixel_values, image_grid_thw)

    # Make modules accessible through properties
    @property
    def language_model(self):
        return self.model.language_model

    @property
    def visual(self):
        return self.model.visual

    # Forward pass through model with input parameters
    # @check_model_inputs
    def forward(self,
                input_ids: torch.LongTensor = None,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_values: Optional[Cache] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                pixel_values: Optional[torch.Tensor] = None,
                pixel_values_videos: Optional[torch.FloatTensor] = None,
                image_grid_thw: Optional[torch.LongTensor] = None,
                video_grid_thw: Optional[torch.LongTensor] = None,
                cache_position: Optional[torch.LongTensor] = None,
                logits_to_keep: Union[int, torch.Tensor] = 0,
                **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, Qwen3VLForEmbeddingOutput]:
        # Pass inputs through the model
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )
        # Return the model output
        return Qwen3VLForEmbeddingOutput(
            last_hidden_state=outputs.last_hidden_state,
            attention_mask=attention_mask,
        )

def sample_frames(frames: List[Union[str, Image.Image]], max_segments: int) -> List[Union[str, Image.Image]]:
    duration = len(frames)
    if duration <= max_segments:
        return frames

    frame_id_array = np.linspace(0, duration - 1, max_segments, dtype=int)
    frame_id_list = frame_id_array.tolist()
    sampled_frames = [ frames[frame_idx] for frame_idx in frame_id_list ]
    return sampled_frames

def is_image_path(path: str) -> bool:
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.svg'}
    
    if path.startswith(('http://', 'https://')):
        # Parse URL to remove query parameters
        parsed_url = urlparse(path)
        clean_path = parsed_url.path
    else:
        clean_path = path
    
    # Check file extension
    _, ext = os.path.splitext(clean_path.lower())
    return ext in image_extensions

def is_video_input(video) -> bool:
    if isinstance(video, str):
        return True
    
    if isinstance(video, list) and len(video) > 0:
        # Check first element to determine the type
        first_elem = video[0]
        
        if isinstance(first_elem, Image.Image):
            return True
        
        if isinstance(first_elem, str):
            return is_image_path(first_elem)
    
    return False

# Define embedder class for processing inputs and generating embeddings
class Qwen3VLEmbedder():
    def __init__(
        self,
        model_name_or_path: str,
        max_length: int = MAX_LENGTH,
        min_pixels: int = MIN_PIXELS,
        max_pixels: int = MAX_PIXELS,
        total_pixels: Optional[int] = None,  # None -> derived from max_length so video tokens fit
        fps: float = FPS,
        max_frames: int = MAX_FRAMES,
        default_instruction: str = "Represent the user's input.",
        revision: Optional[str] = None,  # hub commit to pin (model + processor); None -> default branch
        **kwargs
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.max_length = max_length
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

        # Video token budget: leave room for the text/template tokens, then size the
        # pixel budget so the sampled video fits under `max_length`. An explicit
        # `total_pixels` (if a caller passes one) is still honoured.
        self.video_token_budget = max(TEMPORAL_PATCH_SIZE * VIDEO_MIN_TOKEN_NUM,
                                      max_length - TEXT_TOKEN_RESERVE)
        self.total_pixels = (
            self._tokens_to_total_pixels(self.video_token_budget)
            if total_pixels is None else total_pixels
        )
        self.fps = fps
        # Clamp frames so that even at the per-frame minimum resolution the token
        # floor cannot overflow the budget (a huge max_frames alone can exceed it).
        self.max_frames = self._clamp_max_frames(max_frames, self.video_token_budget)

        self.default_instruction = default_instruction

        # revision pins both the model and the processor to the same hub commit so the
        # whole snapshot (weights + config + tokenizer/processor) is reproducible.
        self.model = Qwen3VLForEmbedding.from_pretrained(
            model_name_or_path, trust_remote_code=True, revision=revision, **kwargs
        ).to(device)
        self.processor = Qwen3VLProcessor.from_pretrained(
            model_name_or_path, padding_side='right', revision=revision
        )
        self.model.eval()

    # --- video token-budget helpers ---------------------------------------
    @staticmethod
    def _tokens_to_total_pixels(tokens: int) -> int:
        """Invert video_tokens ~= total_pixels / (IMAGE_FACTOR**2 * TEMPORAL_PATCH_SIZE)
        to turn a token budget into the `total_pixels` budget qwen_vl_utils consumes."""
        return int(tokens * IMAGE_FACTOR * IMAGE_FACTOR * TEMPORAL_PATCH_SIZE * PIXEL_SAFETY)

    @staticmethod
    def _clamp_max_frames(max_frames: int, token_budget: int) -> int:
        """Cap the frame count so the per-frame minimum (VIDEO_MIN_TOKEN_NUM tokens,
        after temporal merge) cannot alone exceed the budget:
            floor_tokens = (nframes / TEMPORAL_PATCH_SIZE) * VIDEO_MIN_TOKEN_NUM <= budget
        The cap is rounded down to TEMPORAL_PATCH_SIZE (qwen_vl_utils' frame factor)."""
        cap = (token_budget * TEMPORAL_PATCH_SIZE) // VIDEO_MIN_TOKEN_NUM
        cap -= cap % TEMPORAL_PATCH_SIZE
        cap = max(TEMPORAL_PATCH_SIZE, cap)
        if max_frames > cap:
            logger.warning(
                f"max_frames={max_frames} can overflow the video token budget "
                f"({token_budget} tokens); limiting to {cap} frames"
            )
            return cap
        return max_frames

    def _degradation_schedule(self):
        """Order to attempt embedding (in decreasing granularity) until the video fits under `max_length`. 
        Each entry is (total_pixels, max_frames_override, is_last_resort).
        `max_frames_override` is None means 'use the per-item / configured max_frames'.
        The last entry (attempt) is a coarse ~min-resolution fallback that fits any reasonable max_length
        so the video is always encoded (as a sparse/degraded vector) rather than skipped."""
        # Coarse fallback: a few frames near the per-frame minimum resolution.
        coarse_total_pixels = self._tokens_to_total_pixels(
            VIDEO_MIN_TOKEN_NUM * max(1, COARSE_MAX_FRAMES // TEMPORAL_PATCH_SIZE)
        )
        return [
            (self.total_pixels, None, False),                                    # normal
            (max(1, self.total_pixels // 2),                                     # degrade
             max(TEMPORAL_PATCH_SIZE, self.max_frames // 2), False),
            (coarse_total_pixels, COARSE_MAX_FRAMES, True),                      # last resort
        ]

    def assert_dtype(self, expected: torch.dtype = torch.bfloat16) -> None: # cc Claude Opus 4.8
        """Verify every model (and score head) parameter is in `expected` dtype."""
        from collections import Counter
 
        counts = Counter(p.dtype for p in self.model.parameters())
        logger.info(f"Parameter dtypes: {dict(counts)}")
        print(f"Parameter dtypes: {dict(counts)}")
 
        bad = {dtype: n for dtype, n in counts.items() if dtype != expected}
        assert not bad, (
            f"Expected all parameters in {expected}, but found other dtypes: {bad}"
        )

    @torch.no_grad()
    def forward(self, inputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        outputs = self.model(**inputs)
        return {
            'last_hidden_state': outputs.last_hidden_state,
            'attention_mask': inputs.get('attention_mask')
        }

    # Truncate token sequence to a specified max length
    def _truncate_tokens(self, token_ids: List[int], max_length: int) -> List[int]:
        if len(token_ids) <= max_length:
            return token_ids

        special_token_ids = set(self.processor.tokenizer.all_special_ids)
        num_special = sum(1 for token_idx in token_ids if token_idx in special_token_ids)
        num_non_special_to_keep = max_length - num_special

        final_token_ids = []
        non_special_kept_count = 0
        # Ensure retention of special tokens while truncating the rest
        for token_idx in token_ids:
            if token_idx in special_token_ids:
                final_token_ids.append(token_idx)
            elif non_special_kept_count < num_non_special_to_keep:
                final_token_ids.append(token_idx)
                non_special_kept_count += 1
        return final_token_ids

    def format_model_input(
        self, 
        text: Optional[Union[List[str], str]] = None,
        image: Optional[Union[List[Union[str, Image.Image]], str, Image.Image]] = None,
        video: Optional[Union[List[Union[str, List[Union[str, Image.Image]]]], str, List[Union[str, Image.Image]]]] = None,
        instruction: Optional[str] = None,
        fps: Optional[float] = None,
        max_frames: Optional[int] = None,
        video_start: Optional[float] = None,
        video_end: Optional[float] = None,
        total_pixels: Optional[int] = None  # None -> self.total_pixels; overridden by the degrade retry
    ) -> List[Dict]:

        # Pixel budget for this call (a degradation retry passes a smaller one).
        total_pixels = self.total_pixels if total_pixels is None else total_pixels

        # Ensure instruction ends with punctuation
        if instruction:
            instruction = instruction.strip()
            if instruction and not unicodedata.category(instruction[-1]).startswith('P'):
                instruction = instruction + '.'

        # Initialize conversation with system prompts
        content = []
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": instruction or self.default_instruction}]},
            {"role": "user", "content": content}
        ]

        # Normalize text input to list
        if text is None:
            texts = []
        elif isinstance(text, str):
            texts = [text]
        else:
            texts = text
        
        # Normalize image input to list
        if image is None:
            images = []
        elif not isinstance(image, list):
            images = [image]
        else:
            images = image
        
        # Normalize video input to list
        if video is None:
            videos = []
        elif is_video_input(video):
            videos = [video]
        else:
            # Assume it's a list of videos
            videos = video

        # Add text, image, or video content to conversation
        if not texts and not images and not videos:
            content.append({'type': 'text', 'text': "NULL"})
            return conversation

        # Process each video
        for vid in videos:
            video_content = None
            # total_pixels bounds the video token count so it fits under max_length;
            # keep it for BOTH input shapes (previously the file-path branch replaced
            # video_kwargs wholesale and dropped it, letting qwen_vl_utils fall back to
            # its 128k-context default and overflow max_length -> token-count mismatch).
            video_kwargs = {'total_pixels': total_pixels}

            if isinstance(vid, list):
                # Video as frame sequence
                video_content = vid
                if self.max_frames is not None:
                    video_content = sample_frames(video_content, self.max_frames)
                video_content = [
                    ('file://' + ele if isinstance(ele, str) else ele)
                    for ele in video_content
                ]
            elif isinstance(vid, str):
                # Video as file path
                video_content = vid if vid.startswith(('http://', 'https://')) else 'file://' + vid
                video_kwargs.update({'fps': fps or self.fps, 'max_frames': max_frames or self.max_frames}) # in addition to 'total_pixels'
                # Optional temporal trimming: qwen_vl_utils reads video_start/video_end
                # (seconds) from the video content element and only decodes/samples
                # frames within that window -> dense sampling of a segment.
                if video_start is not None:
                    video_kwargs['video_start'] = video_start
                if video_end is not None:
                    video_kwargs['video_end'] = video_end
            else:
                raise TypeError(f"Unrecognized video type: {type(vid)}")

            # Add video input to content
            if video_content:
                content.append({
                    'type': 'video', 
                    'video': video_content,
                    **video_kwargs
                })

        # Process each image
        for img in images:
            image_content = None
            
            if isinstance(img, Image.Image):
                image_content = img
            elif isinstance(img, str):
                image_content = img if img.startswith(('http://', 'https://')) else 'file://' + img
            else:
                raise TypeError(f"Unrecognized image type: {type(img)}")

            # Add image input to content
            if image_content:
                content.append({
                    'type': 'image', 
                    'image': image_content,
                    "min_pixels": self.min_pixels,
                    "max_pixels": self.max_pixels
                })

        # Process each text
        for txt in texts:
            content.append({'type': 'text', 'text': txt})

        return conversation

    # Preprocess input conversations for model consumption
    def _preprocess_inputs(self, conversations: List[List[Dict]]) -> Dict[str, torch.Tensor]:
        text = self.processor.apply_chat_template(
            conversations, add_generation_prompt=True, tokenize=False
        )

        try:
            images, video_inputs, video_kwargs = process_vision_info(
                conversations, image_patch_size=16,
                return_video_metadata=True, return_video_kwargs=True
            )
        except Exception as e:
            logger.error(f"Error in processing vision info: {e}")
            images = None
            video_inputs = None
            video_kwargs = {'do_sample_frames': False}
            text = self.processor.apply_chat_template(
                [{'role': 'user', 'content': [{'type': 'text', 'text': 'NULL'}]}], 
                add_generation_prompt=True, tokenize=False
            )

        if video_inputs is not None:
            videos, video_metadata = zip(*video_inputs)
            videos = list(videos)
            video_metadata = list(video_metadata)
        else:
            videos, video_metadata = None, None

        inputs = self.processor(
            text=text, images=images, videos=videos, video_metadata=video_metadata, truncation=True, 
            max_length=self.max_length, padding=True, do_resize=False, return_tensors='pt',
            **video_kwargs
        )
        return inputs

    # Pool the last hidden state by attention mask for embeddings
    @staticmethod
    def _pooling_last(hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        flipped_tensor = attention_mask.flip(dims=[1])
        last_one_positions = flipped_tensor.argmax(dim=1)
        col = attention_mask.shape[1] - last_one_positions - 1
        row = torch.arange(hidden_state.shape[0], device=hidden_state.device)
        return hidden_state[row, col]

    # Process inputs to generate normalized embeddings
    def process(self, inputs: List[Dict[str, Any]], normalize: bool = True) -> tuple:
        # Try the normal budget first, then progressively coarser video sampling. This
        # guarantees a (possibly sparse/degraded) vector instead of aborting on the
        # processor's "Mismatch in `video` token count ..." error, which fires when a
        # long/high-res video would exceed max_length and truncation clips its tokens.
        # (Unless it is another error, in which case still abort.)
        last_err: Optional[Exception] = None
        schedule = self._degradation_schedule()
        for total_pixels, max_frames_override, is_last_resort in schedule:
            if is_last_resort:
                logger.warning(
                    "video did not fit under max_length=%s at normal or degraded settings; "
                    "input parameters are pathological -- falling back to a coarse last-resort "
                    "encoding (%s frames, total_pixels=%s). The resulting vector is a sparse, "
                    "degraded representation of the video.",
                    self.max_length, max_frames_override, total_pixels,
                )
            try:
                conversations = [self.format_model_input(
                    text=ele.get('text'),
                    image=ele.get('image'),
                    video=ele.get('video'),
                    instruction=ele.get('instruction'),
                    fps=ele.get('fps'),
                    # override the frame cap on degraded retries; else honour the item's
                    max_frames=(max_frames_override if max_frames_override is not None
                                else ele.get('max_frames')),
                    video_start=ele.get('video_start'),
                    video_end=ele.get('video_end'),
                    total_pixels=total_pixels,
                ) for ele in inputs]

                processed_inputs = self._preprocess_inputs(conversations)
                processed_inputs = {k: v.to(self.model.device) for k, v in processed_inputs.items()}

                outputs = self.forward(processed_inputs)
                embeddings = self._pooling_last(outputs['last_hidden_state'], outputs['attention_mask'])

                # Normalize the embeddings if specified
                if normalize:
                    embeddings = F.normalize(embeddings, p=2, dim=-1)

                logger.info("embedding successful")
                return embeddings
            except ValueError as e:
                # Only the video/text token-count mismatch is retryable by shrinking the
                # video; any other ValueError is a real error and must propagate.
                if "Mismatch in" not in str(e):
                    raise
                last_err = e
                logger.warning(
                    f"video token-count mismatch at total_pixels={total_pixels}, "
                    f"max_frames={max_frames_override}; retrying with a smaller budget" # to handle error '{e}'
                )

        # Even the coarse fallback could not fit -- genuinely broken input.
        raise RuntimeError(
            f"could not fit video within max_length={self.max_length} even at the coarse "
            f"last-resort budget"
        ) from last_err
