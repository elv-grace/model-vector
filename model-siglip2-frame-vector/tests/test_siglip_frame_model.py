import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")  # the fakes return torch tensors

from common_ml.tagging.models.tag_types import FrameTag
from common_ml.tagging.models.frame_based import FrameModel

from siglip_frame.model import QUERY_MODES, FeatureExtractor, _WHOLE_FRAME_BOX
from siglip_frame.config import RuntimeConfig

DIM = 768  # base pooled embedding width


def _img() -> np.ndarray:
    return np.zeros((8, 8, 3), dtype=np.uint8)


class _FakeVisionModel:
    """Stands in for the SigLIP 2 vision tower: records the inputs it was called with and
    returns `pooled` as pooler_output. `self.model` *is* the tower, since the production
    path loads Siglip2VisionModel rather than the full dual encoder."""

    def __init__(self, pooled: "torch.Tensor") -> None:
        self.pooled = pooled
        self.seen = None

    def __call__(self, **inputs):
        self.seen = inputs
        return types.SimpleNamespace(pooler_output=self.pooled)


def _make(cfg: RuntimeConfig, pooled: "torch.Tensor") -> FeatureExtractor:
    """Build a FeatureExtractor without running __init__ (no model download), with a fake
    processor and a fake vision tower whose pooler_output is `pooled` (B, DIM)."""
    m = object.__new__(FeatureExtractor)
    m.config = cfg
    m.device = torch.device("cpu")
    m.dtype = torch.float32
    # The real NaFlex processor emits these three tensors; shapes are plausible but the
    # contents don't matter because the vision tower is faked.
    n = cfg.max_num_patches
    m.processor = lambda images, return_tensors="pt", max_num_patches=None: {
        "pixel_values": torch.zeros(1, max_num_patches or n, 768),
        "pixel_attention_mask": torch.ones(1, max_num_patches or n, dtype=torch.long),
        "spatial_shapes": torch.tensor([[8, 8]], dtype=torch.long),
    }
    m.model = _FakeVisionModel(pooled)
    # set by the real __init__ after the load, and stamped into every tag
    m.model_id = "fake/siglip2"
    m.revision = "cafebabe"
    m.dim = int(pooled.shape[-1])
    return m


# ---------------------------------------------------------------------------
# tag_frame / _embed_frame: normalize, box, output shape.
# ---------------------------------------------------------------------------

def test_is_frame_model():
    m = _make(RuntimeConfig(), torch.zeros(1, DIM))
    assert isinstance(m, FrameModel)


def test_tag_frame_returns_single_whole_frame_vector():
    unit = torch.zeros(1, DIM)
    unit[..., 0] = 1.0  # basis vector e_0, already unit length -> normalize is a no-op
    m = _make(RuntimeConfig(normalize=True), unit)

    out = m.tag_frame(_img())

    assert isinstance(out, list) and len(out) == 1
    fv = out[0]
    assert isinstance(fv, FrameTag) and fv.vector is not None
    assert fv.box == _WHOLE_FRAME_BOX
    assert len(fv.vector) == DIM
    assert fv.vector[0] == pytest.approx(1.0)
    assert fv.vector[1] == pytest.approx(0.0)


def test_normalize_true_yields_unit_vector():
    # the pooling head's output is not unit length; normalize=True must fix that
    m = _make(RuntimeConfig(normalize=True), torch.full((1, DIM), 0.3))

    vec = m.tag_frame(_img())[0].vector

    assert float(np.linalg.norm(vec)) == pytest.approx(1.0, abs=1e-5)


def test_normalize_false_keeps_raw_pooled_output():
    m = _make(RuntimeConfig(normalize=False), torch.full((1, DIM), 0.3))

    vec = m.tag_frame(_img())[0].vector

    assert np.allclose(vec, 0.3, atol=1e-6)
    assert float(np.linalg.norm(vec)) == pytest.approx(0.3 * np.sqrt(DIM), abs=1e-4)


# ---------------------------------------------------------------------------
# additional_info: the embedding recipe, so a query can be embedded to match.
# ---------------------------------------------------------------------------

def test_tag_carries_the_recipe_a_query_must_match():
    cfg = RuntimeConfig(normalize=True, max_num_patches=576)
    info = _make(cfg, torch.zeros(1, DIM)).tag_frame(_img())[0].additional_info

    # the exact hub snapshot: the text tower of THIS checkpoint embeds a text query
    assert info["embedder"] == "fake/siglip2"
    assert info["revision"] == "cafebabe"
    # read off the checkpoint, not hardcoded
    assert info["dim"] == DIM
    # whether cosine reduces to a dot product
    assert info["normalize"] is True
    # the NaFlex budget an image query has to be preprocessed at
    assert info["max_num_patches"] == 576


def test_normalize_false_is_recorded_not_assumed():
    # a raw-feature index and a normalized one are different spaces; the flag has to
    # travel with the vector or a query silently normalizes against unnormalized rows
    info = _make(RuntimeConfig(normalize=False), torch.zeros(1, DIM)).tag_frame(_img())[0].additional_info
    assert info["normalize"] is False


def test_tag_declares_its_own_kind_and_the_supported_query_modes():
    info = _make(RuntimeConfig(), torch.zeros(1, DIM)).tag_frame(_img())[0].additional_info

    # what the vector IS -- read directly rather than inferred from which positional
    # fields are populated
    assert info["kind"] == "frame"
    # what a query MAY be: this checkpoint is a dual encoder, so text and image both
    # land in the frame vectors' space; it has no video path
    assert info["query_modes"] == ["text", "image"]
    assert "video" not in info["query_modes"]


def test_tag_owns_its_query_modes_list():
    # a copy, so a consumer mutating one tag's list cannot move the module constant
    info = _make(RuntimeConfig(), torch.zeros(1, DIM)).tag_frame(_img())[0].additional_info
    assert info["query_modes"] == QUERY_MODES
    assert info["query_modes"] is not QUERY_MODES


def test_tag_owns_its_box():
    # each emitted tag must carry its own box dict, not a shared reference to the constant
    m = _make(RuntimeConfig(), torch.zeros(1, DIM))
    fv = m.tag_frame(_img())[0]
    assert fv.box == _WHOLE_FRAME_BOX
    assert fv.box is not _WHOLE_FRAME_BOX


# ---------------------------------------------------------------------------
# _preprocess: NaFlex inputs reach the vision tower, dtypes cast selectively.
# ---------------------------------------------------------------------------

def test_all_naflex_inputs_forwarded_to_vision_tower():
    m = _make(RuntimeConfig(), torch.zeros(1, DIM))

    m.tag_frame(_img())

    seen = m.model.seen
    # the tower needs all three: dropping the mask or the grid silently changes the output
    assert set(seen) == {"pixel_values", "pixel_attention_mask", "spatial_shapes"}


def test_only_pixel_values_cast_to_model_dtype():
    # pixel_attention_mask / spatial_shapes are integer bookkeeping and must keep their
    # dtypes; a blanket .to(dtype) would corrupt the position-embedding interpolation.
    m = _make(RuntimeConfig(), torch.zeros(1, DIM))
    m.dtype = torch.float16

    m.tag_frame(_img())

    seen = m.model.seen
    assert seen["pixel_values"].dtype == torch.float16
    assert seen["pixel_attention_mask"].dtype == torch.long
    assert seen["spatial_shapes"].dtype == torch.long


def test_max_num_patches_threaded_to_processor():
    m = _make(RuntimeConfig(max_num_patches=1024), torch.zeros(1, DIM))

    m.tag_frame(_img())

    # the fake processor sizes pixel_values by the budget it was handed
    assert m.model.seen["pixel_values"].shape[1] == 1024


# ---------------------------------------------------------------------------
# constructor guard
# ---------------------------------------------------------------------------

def test_bad_max_num_patches_raises_before_loading():
    # the guard fires before any weight download, so this needs no model
    with pytest.raises(ValueError):
        FeatureExtractor(cfg=RuntimeConfig(max_num_patches=0))


# ---------------------------------------------------------------------------
# revision pinning: threaded into both from_pretrained calls (no weights loaded)
# ---------------------------------------------------------------------------

def _patch_from_pretrained(monkeypatch, dim: int = DIM) -> dict:
    """Replace Siglip2ImageProcessor / Siglip2VisionModel with fakes that record the
    `revision` passed to from_pretrained, so the real __init__ runs without a download.
    `dim` is the hidden_size the fake checkpoint reports."""
    captured = {}

    class _FakeProcessor:
        @classmethod
        def from_pretrained(cls, model_id, revision=None):
            captured["processor_revision"] = revision
            return cls()

    class _FakeModel:
        # __init__ reads the emitted width off the loaded checkpoint's config
        config = types.SimpleNamespace(hidden_size=dim)

        @classmethod
        def from_pretrained(cls, model_id, revision=None, dtype=None):
            captured["model_revision"] = revision
            return cls()

        def to(self, device):
            return self

        def eval(self):
            return self

    monkeypatch.setattr("siglip_frame.model.Siglip2ImageProcessor", _FakeProcessor)
    monkeypatch.setattr("siglip_frame.model.Siglip2VisionModel", _FakeModel)
    return captured


def test_revision_threaded_to_model_and_processor(monkeypatch):
    captured = _patch_from_pretrained(monkeypatch)

    FeatureExtractor(
        cfg=RuntimeConfig(), model_id="google/siglip2-base-patch16-naflex", revision="cafebabe"
    )

    assert captured["processor_revision"] == "cafebabe"
    assert captured["model_revision"] == "cafebabe"


def test_revision_defaults_to_none(monkeypatch):
    captured = _patch_from_pretrained(monkeypatch)

    FeatureExtractor(cfg=RuntimeConfig(), model_id="google/siglip2-base-patch16-naflex")

    assert captured["processor_revision"] is None
    assert captured["model_revision"] is None


def test_stamped_dim_is_read_from_the_checkpoint(monkeypatch):
    # the width comes off the loaded config, so swapping in -so400m (1152) is reported
    # rather than silently stamped as the -base 768
    _patch_from_pretrained(monkeypatch, dim=1152)

    m = FeatureExtractor(cfg=RuntimeConfig(), model_id="google/siglip2-so400m-patch16-naflex")

    assert m._embedder_info()["dim"] == 1152


def test_stamped_embedder_and_revision_are_what_was_loaded(monkeypatch):
    _patch_from_pretrained(monkeypatch)

    m = FeatureExtractor(
        cfg=RuntimeConfig(), model_id="google/siglip2-base-patch16-naflex", revision="cafebabe"
    )

    info = m._embedder_info()
    assert info["embedder"] == "google/siglip2-base-patch16-naflex"
    assert info["revision"] == "cafebabe"
