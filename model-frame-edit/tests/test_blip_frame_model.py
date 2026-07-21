import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")  # the fakes return torch tensors

from common_ml.tagging.models.tag_types import FrameVector
from common_ml.tagging.models.frame_based import FrameModel

from blip_frame.model import FeatureExtractor, _WHOLE_FRAME_BOX
from blip_frame.config import RuntimeConfig


def _img() -> np.ndarray:
    return np.zeros((8, 8, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# tag_frame / _embed_frame: pooling, normalize, box, output shape.
# _project_tokens is stubbed so no BLIP-2 weights are loaded.
# ---------------------------------------------------------------------------

def _make(cfg: RuntimeConfig, tokens: "torch.Tensor") -> FeatureExtractor:
    """Build a FeatureExtractor without running __init__ (no model download), with a
    fake processor and a stubbed _project_tokens that yields `tokens` (B, 32, 256)."""
    m = object.__new__(FeatureExtractor)
    m.config = cfg
    m.device = torch.device("cpu")
    m.dtype = torch.float32
    # processor(images=..., return_tensors=...).pixel_values -> a throwaway tensor
    # (contents don't matter because _project_tokens is stubbed).
    m.processor = lambda images, return_tensors="pt": types.SimpleNamespace(
        pixel_values=torch.zeros(1, 3, 8, 8)
    )
    m._project_tokens = lambda pixel_values: tokens
    return m


def test_is_frame_model():
    m = _make(RuntimeConfig(), torch.zeros(1, 32, 256))
    assert isinstance(m, FrameModel)


def test_tag_frame_returns_single_whole_frame_vector():
    # 32 identical unit tokens -> mean is that same unit vector, normalize is a no-op.
    unit = torch.zeros(1, 32, 256)
    unit[..., 0] = 1.0  # basis vector e_0, already unit length
    m = _make(RuntimeConfig(normalize=True), unit)

    out = m.tag_frame(_img())

    assert isinstance(out, list) and len(out) == 1
    fv = out[0]
    assert isinstance(fv, FrameVector)
    assert fv.box == _WHOLE_FRAME_BOX
    assert len(fv.vector) == 256
    assert fv.vector[0] == pytest.approx(1.0)
    assert fv.vector[1] == pytest.approx(0.0)


def test_normalize_true_yields_unit_vector():
    # tokens whose mean is NOT unit length; normalize=True must renormalize to 1.
    tokens = torch.full((1, 32, 256), 0.3)
    m = _make(RuntimeConfig(normalize=True), tokens)

    vec = m.tag_frame(_img())[0].vector

    assert float(np.linalg.norm(vec)) == pytest.approx(1.0, abs=1e-5)


def test_normalize_false_keeps_raw_mean():
    # normalize=False emits the raw pooled mean (not renormalized).
    tokens = torch.full((1, 32, 256), 0.3)
    m = _make(RuntimeConfig(normalize=False), tokens)

    vec = m.tag_frame(_img())[0].vector

    # mean over the 32 identical tokens is 0.3 in every dim
    assert np.allclose(vec, 0.3, atol=1e-6)
    # and that vector is clearly not unit length (0.3 * sqrt(256) = 4.8)
    assert float(np.linalg.norm(vec)) == pytest.approx(0.3 * np.sqrt(256), abs=1e-4)


def test_tag_owns_its_box():
    # each emitted tag must carry its own box dict, not a shared reference to the constant
    m = _make(RuntimeConfig(), torch.zeros(1, 32, 256))
    fv = m.tag_frame(_img())[0]
    assert fv.box == _WHOLE_FRAME_BOX
    assert fv.box is not _WHOLE_FRAME_BOX


# ---------------------------------------------------------------------------
# _project_tokens: ViT -> Q-Former -> vision projection wiring, with a fake model.
# ---------------------------------------------------------------------------

class _FakeQFormer:
    def __call__(self, query_embeds, encoder_hidden_states, encoder_attention_mask, return_dict):
        # a real Q-Former mixes image features into the queries; the fake just passes the
        # (B, 32, 768) query tokens through so the projection/normalize can be checked.
        assert return_dict is True
        return types.SimpleNamespace(last_hidden_state=query_embeds)


class _FakeVisionModel:
    def __call__(self, pixel_values, return_dict):
        assert return_dict is True
        b = pixel_values.shape[0]
        # (B, num_patches, vision_hidden); values irrelevant (qformer is faked)
        return types.SimpleNamespace(last_hidden_state=torch.randn(b, 10, 768))


class _FakeModel:
    def __init__(self):
        self.vision_model = _FakeVisionModel()
        self.qformer = _FakeQFormer()
        self.query_tokens = torch.randn(1, 32, 768)
        # a real Linear so .weight.dtype exists and the projection actually runs
        self.vision_projection = torch.nn.Linear(768, 256)


def test_project_tokens_shape_and_per_token_normalized():
    m = object.__new__(FeatureExtractor)
    m.model = _FakeModel()

    tokens = m._project_tokens(torch.zeros(1, 3, 8, 8))

    assert tokens.shape == (1, 32, 256)
    # each of the 32 tokens is L2-normalized along the last dim
    norms = tokens.norm(p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_project_tokens_batched():
    # wiring stays correct for B > 1 (query tokens expanded per image)
    m = object.__new__(FeatureExtractor)
    m.model = _FakeModel()

    tokens = m._project_tokens(torch.zeros(4, 3, 8, 8))

    assert tokens.shape == (4, 32, 256)


# ---------------------------------------------------------------------------
# constructor guard
# ---------------------------------------------------------------------------

def test_unsupported_pooling_raises_before_loading():
    # the guard fires before any weight download, so this needs no model
    with pytest.raises(NotImplementedError):
        FeatureExtractor(cfg=RuntimeConfig(pooling="tokens"))
