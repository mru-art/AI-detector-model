"""End-to-end test of the distillation loop on a tiny synthetic dataset.

Runs with random ViT-B weights (no download), so it checks the mechanics, not accuracy:
memory is built, the loop adds each pending generator, the memory is reused on resume,
and nothing is retrained from scratch.

    python -m pytest distill_loop/test_loop.py -q
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.dirname(ROOT)]

import numpy as np
import pytest
import torch
from PIL import Image

from cl_common import config as cfg
from cl_common.evaluation import evaluate_seen
from cl_common.runner import CLError, save_results
from distill_loop import loop
from distill_loop.memory import ExemplarMemory
from models import create_model

CLASSES = ["real", "dalle3", "midjourney"]


def _make_split(root, split, n):
    rng = np.random.default_rng(0 if split == "train" else 1)
    for c in CLASSES:
        d = os.path.join(root, split, c)
        os.makedirs(d, exist_ok=True)
        for i in range(n):
            arr = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
            Image.fromarray(arr).save(os.path.join(d, f"{i}.png"))


@pytest.fixture
def env(tmp_path, monkeypatch):
    data = tmp_path / "dataset"
    _make_split(str(data), "train", 6)
    _make_split(str(data), "test", 3)
    monkeypatch.setattr(cfg, "DATA_DIR", str(data))
    monkeypatch.setattr(cfg, "RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setattr(cfg, "GENERATORS", ["dalle3", "midjourney"])
    monkeypatch.setattr(cfg, "BATCH_SIZE", 4)
    monkeypatch.setattr(cfg, "NUM_WORKERS", 0)
    monkeypatch.setattr(cfg, "HEAD_EPOCHS", 1)
    monkeypatch.setattr(cfg, "DISTILL_EPOCHS", 1)
    monkeypatch.setattr(cfg, "DISTILL_EXEMPLARS_PER_CLASS", 2)
    monkeypatch.setattr(cfg, "EVAL_TTA", False)
    monkeypatch.setattr(cfg, "DEVICE", "cpu")
    return tmp_path


def test_memory_roundtrip_and_no_overwrite(tmp_path):
    m = ExemplarMemory(per_class=2)
    m.ensure("real", ["a", "b", "c", "d"], seed=1)
    first = list(m.items["real"])
    m.ensure("real", ["x", "y"], seed=1)           # existing class is not reselected
    assert m.items["real"] == first and len(first) == 2
    path = tmp_path / "ex.json"
    m.save(str(path))
    assert ExemplarMemory.load(str(path), 2).items == m.items


def test_loop_adds_pending_generators_and_saves_memory(env, monkeypatch):
    # source model has learned only the first generator; the loop must add the second one
    monkeypatch.setattr(cfg, "GENERATORS", ["dalle3", "midjourney"])
    torch.manual_seed(0)
    src = create_model(cfg.NUM_CLASSES, pretrained=False)
    hist = [evaluate_seen(src, 0)]
    save_results("src", src, hist)

    metrics = loop.run_update("src")

    out = env / "results" / "distill_loop"
    assert (out / "final_model.pth").exists()
    assert (out / "exemplars.json").exists()
    assert metrics["generators"] == ["dalle3", "midjourney"]
    mem = ExemplarMemory.load(str(out / "exemplars.json"), 2)
    assert set(mem.items) == {"real", "dalle3", "midjourney"}
    assert all(len(v) == 2 for v in mem.items.values())


def test_distillation_term_is_active_on_new_generators(env, monkeypatch):
    """KD must be used on every step after the first generator's teacher exists."""
    from kd.strategy import KD
    monkeypatch.setattr(cfg, "GENERATORS", ["dalle3", "midjourney"])
    torch.manual_seed(0)
    src = create_model(cfg.NUM_CLASSES, pretrained=False)
    save_results("src", src, [evaluate_seen(src, 0)])

    seen = []
    original = KD.loss

    def spy(self, model, x, y):
        seen.append(self.teacher is not None and bool(self.old_classes))
        return original(self, model, x, y)

    monkeypatch.setattr(KD, "loss", spy)
    loop.run_update("src")
    assert seen and all(seen)


def test_resume_reuses_saved_memory_and_nothing_pending(env, monkeypatch):
    monkeypatch.setattr(cfg, "GENERATORS", ["dalle3", "midjourney"])
    torch.manual_seed(0)
    src = create_model(cfg.NUM_CLASSES, pretrained=False)
    save_results("src", src, [evaluate_seen(src, 0)])
    loop.run_update("src")

    out = env / "results" / "distill_loop"
    before = ExemplarMemory.load(str(out / "exemplars.json"), 2).items

    with pytest.raises(CLError, match="Nothing to add"):
        loop.run_update("src")
    assert ExemplarMemory.load(str(out / "exemplars.json"), 2).items == before


def test_missing_source_is_a_clear_error(env):
    with pytest.raises(CLError, match="No trained model"):
        loop.run_update("does_not_exist")
