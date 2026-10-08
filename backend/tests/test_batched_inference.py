"""
Batched model inference must give the same answers as one-at-a-time inference.

CLIP attribute crops are recorded in a dry run and encoded together; BLIP
keyframe captions are generated in batches. The models are faked here so these
tests check the batching mechanics, not model weights (both were verified
against per-item inference on real footage: identical attribute values and
identical captions).
"""
import numpy as np
import pytest
import torch

from app.core.config import settings
from app.services.pipeline import clip_attributes as C
from app.services.semantic.vlm_service import VisionLanguageService as VLS

LABELS = (["red", "green", "blue"], ["a red thing", "a green thing", "a blue thing"])


class _FakeClip:
    logit_scale = torch.nn.Parameter(torch.tensor(float(np.log(100.0))))


@pytest.fixture
def fake_clip(monkeypatch):
    calls = []

    def fake_encode(crops):
        calls.append(len(crops))
        f = torch.tensor([[float(c[..., ch].mean()) + 1.0 for ch in (2, 1, 0)] for c in crops])
        return f / f.norm(dim=-1, keepdim=True)

    monkeypatch.setattr(C, "_encode", fake_encode)
    monkeypatch.setattr(C, "_model", lambda: (_FakeClip(), None))
    monkeypatch.setattr(C, "_text_features", lambda prompts: torch.eye(3)[: len(prompts)])
    monkeypatch.setattr(settings, "ENABLE_ATTRIBUTE_CLASSIFICATION", True)
    return calls


def _crops():
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (200, 300, 3), dtype=np.uint8)
    return [frame[10:60, 10:80], frame[50:150, 100:200], frame[10:60, 10:80], frame[120:190, 200:290]]


def test_recording_collects_crops_without_running_clip(fake_clip):
    crops = _crops()
    with C.recording() as recorded:
        assert all(C.classify(c, {"q": LABELS}) == {} for c in crops)
    assert len(recorded) == 4 and fake_clip == []


def test_served_batch_matches_per_crop_classification(fake_clip):
    crops = _crops()
    expected = [C.classify(c, {"q": LABELS}) for c in crops]
    fake_clip.clear()

    with C.recording() as recorded:
        for c in crops:
            C.classify(c, {"q": LABELS})
    features = C.encode_batch(recorded)
    assert fake_clip == [3]          # one forward pass; the repeated crop is encoded once
    with C.serving(features):
        got = [C.classify(c, {"q": LABELS}) for c in crops]
    assert fake_clip == [3]          # every lookup was served from the batch
    for e, g in zip(expected, got):
        assert e.keys() == g.keys()
        for label in e["q"]:
            assert g["q"][label] == pytest.approx(e["q"][label], abs=1e-6)


def test_served_miss_falls_back_to_per_crop(fake_clip):
    crops = _crops()
    with C.serving(C.encode_batch(crops[:1])):
        assert C.classify(crops[1], {"q": LABELS})["q"]
    assert fake_clip == [1, 1]       # batch of one, then a per-crop encode for the miss


def test_nothing_is_recorded_when_attribute_classification_is_disabled(fake_clip, monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_ATTRIBUTE_CLASSIFICATION", False)
    with C.recording() as recorded:
        C.classify(_crops()[0], {"q": LABELS})
    assert recorded == []


def test_caption_batch_failure_retries_images_individually(monkeypatch):
    def fake_batch(paths):
        if len(paths) > 1 or paths[0] == "bad.jpg":
            raise RuntimeError("cannot decode")
        return [f"caption of {paths[0]}"]

    monkeypatch.setattr(VLS, "_caption_batch", classmethod(lambda cls, paths: fake_batch(paths)))
    got = VLS._caption_many([(1, "a.jpg"), (2, "bad.jpg"), (3, "c.jpg")])
    assert got == {1: "caption of a.jpg", 3: "caption of c.jpg"}   # only the bad file loses its caption


def test_captions_are_batched(monkeypatch):
    seen = []

    def fake_batch(cls, paths):
        seen.append(len(paths))
        return [f"caption of {p}" for p in paths]

    monkeypatch.setattr(VLS, "_caption_batch", classmethod(fake_batch))
    items = [(i, f"{i}.jpg") for i in range(VLS.CAPTION_BATCH_SIZE + 3)]
    got = VLS._caption_many(items)
    assert seen == [VLS.CAPTION_BATCH_SIZE, 3]
    assert got == {i: f"caption of {i}.jpg" for i in range(VLS.CAPTION_BATCH_SIZE + 3)}
