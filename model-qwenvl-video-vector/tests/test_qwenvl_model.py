import os
import sys
import types

import numpy as np
import pytest

from common_ml.tagging.messages import Tag
from common_ml.tagging.models.av import AVModel

# Mock the embedder so it never touches the real Qwen3VLEmbedder for speed
# because of the heavy transformers/qwen_vl_utils stack imported by `embedding.model`
# at load time.
# Stub the submodule so `embedding.model` still imports when the stack isn't installed
# (e.g. a lightweight host env), but skips the end-to-end test.
# In the container, the real module is loaded and the end-to-end test runs on the real path.
try:
    import transformers  # noqa: F401
except ImportError:
    _stub = types.ModuleType("embedding.qwen3_vl_embedding")
    _stub.Qwen3VLEmbedder = type("Qwen3VLEmbedder", (), {})
    sys.modules["embedding.qwen3_vl_embedding"] = _stub

from embedding.model import QwenVLVideoEmbedder

# The video token-budget / degradation logic lives on the real Qwen3VLEmbedder; when the
# transformers stack is absent the submodule is stubbed above, so those tests are skipped.
import embedding.qwen3_vl_embedding as qmod  # noqa: E402
_REAL_EMBEDDER = hasattr(qmod.Qwen3VLEmbedder, "_degradation_schedule")
requires_real_embedder = pytest.mark.skipif(
    not _REAL_EMBEDDER, reason="real Qwen3VLEmbedder (transformers stack) not importable"
)

# Optional end-to-end config: set these to exercise the real Qwen3-VL model.
E2E_MODEL = os.environ.get("QWENVL_EMBEDDER_PATH")
E2E_VIDEO = os.environ.get("QWENVL_TEST_VIDEO")


class _FakeEmbedder:
    """Stands in for Qwen3VLEmbedder: records the item dicts passed to process()
    and returns a deterministic torch tensor, so the tagger's window conversion
    and wiring can be tested without loading the 8B model."""

    def __init__(self, dim: int = 4):
        self.dim = dim
        self.calls = []

    def process(self, inputs, normalize=True):
        import torch
        self.calls.append({"inputs": inputs, "normalize": normalize})
        # one row per input
        return torch.arange(len(inputs) * self.dim, dtype=torch.float32).reshape(len(inputs), self.dim)


def _make_tagger_with_fake(fake, prompt=None, normalize=None, segment_length_s=None) -> QwenVLVideoEmbedder:
    # bypass __init__ so no model is loaded
    tagger = object.__new__(QwenVLVideoEmbedder)
    tagger.embedder = fake
    tagger.fps = 1.0
    tagger.max_frames = 64
    tagger.prompt = prompt
    tagger.normalize = normalize
    tagger.segment_length_s = segment_length_s
    return tagger


torch = pytest.importorskip("torch")  # the fake returns a torch tensor


def test_is_av_model():
    # the embedder is itself an AVModel, so run_default's AVModel branch consumes it
    tagger = _make_tagger_with_fake(_FakeEmbedder())
    assert isinstance(tagger, AVModel)


def test_embed_video_converts_ms_to_seconds():
    fake = _FakeEmbedder()
    tagger = _make_tagger_with_fake(fake)

    vec = tagger._embed_video("v.mp4", start_ms=30_000, end_ms=45_500)

    assert len(fake.calls) == 1
    item = fake.calls[0]["inputs"][0]
    assert item["video"] == "v.mp4"
    # ms -> seconds for Qwen's native video reader
    assert item["video_start"] == 30.0
    assert item["video_end"] == 45.5
    # self.normalize is None -> the embedder's default (L2-normalize)
    assert fake.calls[0]["normalize"] is True
    # returns a plain python list
    assert isinstance(vec, list)
    assert vec == [0.0, 1.0, 2.0, 3.0]


def test_embed_video_whole_media_omits_window():
    fake = _FakeEmbedder()
    tagger = _make_tagger_with_fake(fake)

    tagger._embed_video("v.mp4", start_ms=None, end_ms=None)

    item = fake.calls[0]["inputs"][0]
    # no trimming keys when the window is the whole media
    assert "video_start" not in item
    assert "video_end" not in item


def test_prompt_forwarded_when_set():
    fake = _FakeEmbedder()
    tagger = _make_tagger_with_fake(fake, prompt="Represent the video.")

    tagger._embed_video("v.mp4")

    assert fake.calls[0]["inputs"][0]["instruction"] == "Represent the video."


def test_embed_video_threads_normalize_flag():
    # self.normalize is threaded down to process()
    fake = _FakeEmbedder()
    tagger = _make_tagger_with_fake(fake, normalize=False)
    tagger._embed_video("v.mp4")
    assert fake.calls[0]["normalize"] is False


def test_embed_video_normalize_none_defaults_true():
    # None -> the embedder's default (L2-normalize)
    fake = _FakeEmbedder()
    tagger = _make_tagger_with_fake(fake, normalize=None)
    tagger._embed_video("v.mp4")
    assert fake.calls[0]["normalize"] is True


def test_tag_whole_video_single_window(monkeypatch):
    # segment_length_s unset -> one whole-video window -> exactly one Tag with vector
    fake = _FakeEmbedder()
    tagger = _make_tagger_with_fake(fake)
    monkeypatch.setattr("embedding.model.get_duration", lambda p: 12.0)

    tags = tagger.tag("v.mp4")

    assert len(tags) == 1
    v = tags[0]
    assert isinstance(v, Tag) and v.vector is not None
    assert v.frame_info is None
    assert v.start_time == 0 # start = end = 0 when just 1 window (whole video)
    assert v.end_time == 0  # because tagger should align when run on full content
    assert len(v.vector) == fake.dim
    # the single window spans [0, duration] and is passed to the embedder in seconds
    item = fake.calls[0]["inputs"][0]
    assert item["video_start"] == 0.0
    assert item["video_end"] == 12.0


def test_tag_segments_emit_one_vector_each(monkeypatch):
    # 25s media / 10s segments -> windows [0,10],[10,20],[20,25], one Tag with vector each
    fake = _FakeEmbedder()
    tagger = _make_tagger_with_fake(fake, segment_length_s=10.0)
    monkeypatch.setattr("embedding.model.get_duration", lambda p: 25.0)

    tags = tagger.tag("v.mp4")

    assert len(tags) == 3
    assert all(isinstance(t, Tag) and t.vector is not None for t in tags)
    assert [(t.start_time, t.end_time) for t in tags] == [(0, 10_000), (10_000, 20_000), (20_000, 25_000)]
    # no pooled whole-video tag is appended
    assert all(t.frame_info is None for t in tags)
    # each window is trimmed in seconds for Qwen's native reader
    windows_s = [(c["inputs"][0].get("video_start"), c["inputs"][0].get("video_end")) for c in fake.calls]
    assert windows_s == [(0.0, 10.0), (10.0, 20.0), (20.0, 25.0)]


def test_tag_bad_segment_is_skipped_not_fatal(monkeypatch):
    # a single failing window must not abort the whole media
    class _FlakyEmbedder(_FakeEmbedder):
        def process(self, inputs, normalize=True):
            if len(self.calls) == 1:  # fail on the second window
                self.calls.append({"inputs": inputs, "normalize": normalize})
                raise RuntimeError("boom")
            return super().process(inputs, normalize=normalize)

    fake = _FlakyEmbedder()
    tagger = _make_tagger_with_fake(fake, segment_length_s=10.0)
    monkeypatch.setattr("embedding.model.get_duration", lambda p: 25.0)

    tags = tagger.tag("v.mp4")  # must NOT raise

    # every window was attempted, but the failed one produced no Tag with vector
    assert len(fake.calls) == 3
    assert len(tags) == 2
    assert [(t.start_time, t.end_time) for t in tags] == [(0, 10_000), (20_000, 25_000)]


# -------------------- dtype auto-selection (constructor) --------------------

def _capture_embedder(monkeypatch):
    """Patch Qwen3VLEmbedder with a stand-in that records its constructor kwargs,
    so the real __init__ (and its dtype logic) runs without loading a model."""
    captured = {}

    class _CapturingEmbedder:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("embedding.model.Qwen3VLEmbedder", _CapturingEmbedder)
    return captured


def test_dtype_autoselects_bfloat16_when_supported(monkeypatch):
    captured = _capture_embedder(monkeypatch)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)

    QwenVLVideoEmbedder(embedder_path="x")

    assert captured["dtype"] is torch.bfloat16


def test_dtype_falls_back_to_float16_without_bf16(monkeypatch):
    # CUDA present but no bf16 support (older GPU) -> float16
    captured = _capture_embedder(monkeypatch)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)

    QwenVLVideoEmbedder(embedder_path="x")

    assert captured["dtype"] is torch.float16


def test_dtype_falls_back_to_float16_without_cuda(monkeypatch):
    # no CUDA -> float16, and is_bf16_supported must be short-circuited (never called)
    captured = _capture_embedder(monkeypatch)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def _boom():
        raise AssertionError("is_bf16_supported must not be called when CUDA is absent")

    monkeypatch.setattr(torch.cuda, "is_bf16_supported", _boom)

    QwenVLVideoEmbedder(embedder_path="x")

    assert captured["dtype"] is torch.float16


def test_explicit_dtype_is_respected(monkeypatch):
    # an explicit dtype wins even on a bf16-capable GPU
    captured = _capture_embedder(monkeypatch)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)

    QwenVLVideoEmbedder(embedder_path="x", dtype=torch.float32)

    assert captured["dtype"] is torch.float32


# -------------------- revision pinning (weights constructor) --------------------

def test_revision_forwarded_to_embedder(monkeypatch):
    # the configured hub commit is threaded down into the embedder (which pins both
    # the model and the processor from_pretrained calls to it)
    captured = _capture_embedder(monkeypatch)

    QwenVLVideoEmbedder(embedder_path="Qwen/Qwen3-VL-Embedding-8B", revision="deadbeef")

    assert captured["revision"] == "deadbeef"


def test_revision_defaults_to_none(monkeypatch):
    # no revision -> None -> the embedder tracks the default branch
    captured = _capture_embedder(monkeypatch)

    QwenVLVideoEmbedder(embedder_path="Qwen/Qwen3-VL-Embedding-8B")

    assert captured.get("revision") is None


# -------------------- tag(): empty / all-failed / collapsed windows --------------------

def test_tag_empty_embedding_emits_nothing(monkeypatch):
    # a window yielding an empty vector is skipped; with no usable vectors,
    # tag() emits nothing rather than a Tag with zero-length vector
    class _EmptyEmbedder(_FakeEmbedder):
        def process(self, inputs, normalize=True):
            self.calls.append({"inputs": inputs, "normalize": normalize})
            return torch.empty((1, 0), dtype=torch.float32)

    fake = _EmptyEmbedder()
    tagger = _make_tagger_with_fake(fake)
    monkeypatch.setattr("embedding.model.get_duration", lambda p: 12.0)

    assert tagger.tag("v.mp4") == []
    assert len(fake.calls) == 1  # the window was still attempted


def test_tag_all_segments_fail_emits_nothing(monkeypatch):
    # every window raises -> no usable vectors -> empty output, no exception
    class _AlwaysFailEmbedder(_FakeEmbedder):
        def process(self, inputs, normalize=True):
            self.calls.append({"inputs": inputs, "normalize": normalize})
            raise RuntimeError("boom")

    fake = _AlwaysFailEmbedder()
    tagger = _make_tagger_with_fake(fake, segment_length_s=10.0)
    monkeypatch.setattr("embedding.model.get_duration", lambda p: 25.0)

    assert tagger.tag("v.mp4") == []
    assert len(fake.calls) == 3  # every window attempted despite failures


def test_tag_subframe_trailing_remainder_folds_into_previous(monkeypatch):
    # 30.03s media / 30s segments at 1 fps: naive windowing yields a 30ms tail
    # [30.0s, 30.03s] shorter than one frame, which qwen_vl_utils rejects as an
    # "Invalid time range". It must be folded into the previous window, leaving a
    # single [0, 30.03s] segment (one Tag with vector).
    fake = _FakeEmbedder()
    tagger = _make_tagger_with_fake(fake, segment_length_s=30.0)  # fps defaults to 1.0
    monkeypatch.setattr("embedding.model.get_duration", lambda p: 30.03)

    tags = tagger.tag("v.mp4")

    # one merged window is embedded (the tail is not a second segment)
    assert len(tags) == 1
    assert len(fake.calls) == 1
    # asserted on the embedder call (the window), since that is what the fold changes:
    # the single window spans the whole media [0, 30.03s].
    item = fake.calls[0]["inputs"][0]
    assert item["video_start"] == 0.0
    assert item["video_end"] == pytest.approx(30.03)


def test_tag_trailing_remainder_at_least_one_frame_is_kept(monkeypatch):
    # a trailing remainder >= one frame period is a genuine segment and must be kept.
    # fps=2 -> one frame = 500ms; 20.6s / 10s segments -> last window [20.0s, 20.6s]
    # is 600ms >= 500ms, so it stays its own segment (not folded).
    fake = _FakeEmbedder()
    tagger = _make_tagger_with_fake(fake, segment_length_s=10.0)
    tagger.fps = 2.0
    monkeypatch.setattr("embedding.model.get_duration", lambda p: 20.6)

    tags = tagger.tag("v.mp4")

    assert len(tags) == 3
    assert len(fake.calls) == 3
    # windows the embedder actually received (in seconds); the tail [20.0, 20.6] survives
    windows_s = [(c["inputs"][0].get("video_start"), c["inputs"][0].get("video_end")) for c in fake.calls]
    assert windows_s[:2] == [(0.0, 10.0), (10.0, 20.0)]
    assert windows_s[2][0] == 20.0
    assert windows_s[2][1] == pytest.approx(20.6)


def test_tag_segment_length_ge_duration_collapses_to_one_window(monkeypatch):
    # segment_length_s >= media duration -> a single whole-video window
    fake = _FakeEmbedder()
    tagger = _make_tagger_with_fake(fake, segment_length_s=60.0)
    monkeypatch.setattr("embedding.model.get_duration", lambda p: 12.0)

    tags = tagger.tag("v.mp4")

    assert len(tags) == 1
    assert (tags[0].start_time, tags[0].end_time) == (0, 0)
    assert len(fake.calls) == 1


# -------------------- end-to-end (opt-in) --------------------

@pytest.mark.skipif(
    not (E2E_MODEL and E2E_VIDEO and torch.cuda.is_available()),
    reason="set QWENVL_EMBEDDER_PATH + QWENVL_TEST_VIDEO and have CUDA to run the real model",
)
def test_end_to_end_one_vector_per_video():
    model = QwenVLVideoEmbedder(embedder_path=E2E_MODEL, fps=1.0, max_frames=64)

    tags = model.tag(E2E_VIDEO)

    # whole-video window (segment_length_s unset) -> exactly one Tag with vector
    assert len(tags) == 1
    v = tags[0]
    assert isinstance(v, Tag) and v.vector is not None
    assert v.frame_info is None
    assert v.start_time == 0
    assert len(v.vector) > 0
    # normalized -- tolerance is loose because the model runs in bfloat16 on
    # bf16-capable GPUs (~2^-8 relative precision), so the norm won't hit 1.0 as
    # tightly as an fp32 run would.
    assert abs(float(np.linalg.norm(v.vector)) - 1.0) < 1e-2  # normalized


# -------------------- video token budget & degradation (embedder) --------------------
# These exercise Qwen3VLEmbedder's fit-to-max_length logic without loading the 8B model,
# by building a bare instance (object.__new__) and setting only the attributes the budget
# code reads.

def _bare_embedder(max_length: int = 8192, max_frames: int = 64) -> "qmod.Qwen3VLEmbedder":
    e = object.__new__(qmod.Qwen3VLEmbedder)
    e.max_length = max_length
    e.min_pixels = qmod.MIN_PIXELS
    e.max_pixels = qmod.MAX_PIXELS
    e.default_instruction = "Represent."
    e.fps = 1.0
    e.video_token_budget = max(
        qmod.TEMPORAL_PATCH_SIZE * qmod.VIDEO_MIN_TOKEN_NUM, max_length - qmod.TEXT_TOKEN_RESERVE
    )
    e.total_pixels = qmod.Qwen3VLEmbedder._tokens_to_total_pixels(e.video_token_budget)
    e.max_frames = qmod.Qwen3VLEmbedder._clamp_max_frames(max_frames, e.video_token_budget)
    return e


def _est_tokens(total_pixels: int) -> float:
    # inverse of _tokens_to_total_pixels, ignoring the safety factor
    return total_pixels / (qmod.IMAGE_FACTOR ** 2 * qmod.TEMPORAL_PATCH_SIZE)


@requires_real_embedder
def test_total_pixels_budget_fits_under_max_length():
    # the derived total_pixels must map back to fewer tokens than the budget
    e = _bare_embedder(max_length=8192)
    assert _est_tokens(e.total_pixels) <= e.video_token_budget
    assert e.video_token_budget < e.max_length  # text headroom reserved


@requires_real_embedder
def test_clamp_max_frames_caps_pathological_and_keeps_reasonable():
    budget = 8192 - qmod.TEXT_TOKEN_RESERVE
    # a reasonable frame count is untouched
    assert qmod.Qwen3VLEmbedder._clamp_max_frames(64, budget) == 64
    # an enormous one is clamped so the per-frame token floor can't overflow the budget
    capped = qmod.Qwen3VLEmbedder._clamp_max_frames(100_000, budget)
    assert capped < 100_000
    assert (capped // qmod.TEMPORAL_PATCH_SIZE) * qmod.VIDEO_MIN_TOKEN_NUM <= budget
    assert capped % qmod.TEMPORAL_PATCH_SIZE == 0


@requires_real_embedder
def test_format_model_input_keeps_total_pixels_for_file_path():
    e = _bare_embedder()
    conv = e.format_model_input(video="v.mp4")
    vc = [c for c in conv[1]["content"] if c["type"] == "video"][0]
    assert vc["total_pixels"] == e.total_pixels
    assert vc["fps"] == e.fps
    assert vc["max_frames"] == e.max_frames


@requires_real_embedder
def test_format_model_input_total_pixels_override():
    e = _bare_embedder()
    conv = e.format_model_input(video="v.mp4", total_pixels=12345)
    vc = [c for c in conv[1]["content"] if c["type"] == "video"][0]
    assert vc["total_pixels"] == 12345


@requires_real_embedder
def test_degradation_schedule_shrinks_and_ends_in_last_resort():
    e = _bare_embedder()
    sched = e._degradation_schedule()
    # strictly shrinking pixel budgets
    pixels = [tp for tp, _, _ in sched]
    assert pixels == sorted(pixels, reverse=True)
    # exactly one last-resort entry, and it is the final one
    assert [is_last for _, _, is_last in sched] == [False, False, True]
    # the coarse fallback easily fits under the budget
    assert _est_tokens(sched[-1][0]) < e.video_token_budget


@requires_real_embedder
def test_process_degrades_on_token_mismatch():
    # first two attempts hit the processor's token-count mismatch; the third succeeds.
    e = _bare_embedder()
    attempts = {"n": 0}

    def fake_pre(conversations):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise ValueError("Mismatch in `video` token count between text and `input_ids`.")
        return {}  # empty -> the device-move comprehension is a no-op

    e._preprocess_inputs = fake_pre
    e.forward = lambda inp: {
        "last_hidden_state": torch.zeros(1, 1, 4),
        "attention_mask": torch.ones(1, 1),
    }

    out = e.process([{"video": "v.mp4"}], normalize=False)

    assert attempts["n"] == 3          # degraded twice, then fit
    assert tuple(out.shape) == (1, 4)  # a (bad/sparse) vector, not an exception


@requires_real_embedder
def test_process_propagates_non_mismatch_valueerror():
    # a ValueError that is NOT the token mismatch is a real error and must propagate
    e = _bare_embedder()

    def fake_pre(conversations):
        raise ValueError("some other problem")

    e._preprocess_inputs = fake_pre
    with pytest.raises(ValueError, match="some other problem"):
        e.process([{"video": "v.mp4"}], normalize=False)


@requires_real_embedder
def test_process_pathological_input_logs_and_raises(caplog):
    # every attempt (including the coarse last resort) mismatches -> pathological input:
    # a warning naming it, then a RuntimeError rather than a silent skip.
    import logging

    e = _bare_embedder()

    def fake_pre(conversations):
        raise ValueError("Mismatch in `video` token count between text and `input_ids`.")

    e._preprocess_inputs = fake_pre
    with caplog.at_level(logging.WARNING, logger=qmod.logger.name):
        with pytest.raises(RuntimeError, match="coarse last-resort"):
            e.process([{"video": "v.mp4"}], normalize=False)

    assert any("pathological" in r.getMessage() for r in caplog.records)
