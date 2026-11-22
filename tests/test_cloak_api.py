"""Tests for POST /cloak. Uses a small random SCRAM-layout checkpoint, so no download is needed."""
import base64
import io

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    from torchvision.models import vit_b_16

    m = vit_b_16(weights=None)
    m.heads.head = nn.Sequential(nn.Linear(768, 512), nn.GELU(), nn.Dropout(0.4), nn.Linear(512, 256),
                                 nn.GELU(), nn.Dropout(0.3), nn.Linear(256, 8))
    ckpt = tmp_path / "scram.pth"
    torch.save({"model_state_dict": m.state_dict()}, ckpt)
    monkeypatch.setenv("MODEL_PATH", str(ckpt))
    monkeypatch.setenv("CLOAK_STEPS", "3")
    from backend.api import cloak as cloak_module
    monkeypatch.setattr(cloak_module, "_encoder", None)
    from backend.api.main import app
    return TestClient(app)              # no 'with' block: the detector is not loaded


def _png(size=(224, 224)):
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "PNG")
    return buf.getvalue()


def test_cloak_returns_image_and_scores(client):
    r = client.post("/cloak", files={"file": ("a.png", _png(), "image/png")}, data={"steps": "3"})
    assert r.status_code == 200, r.text
    body = r.json()
    img = Image.open(io.BytesIO(base64.b64decode(body["image_png_base64"])))
    assert img.size == (224, 224)
    assert 0.0 < body["ssim"] <= 1.0
    assert 0.0 <= body["health"] <= 100.0
    assert "Identity" in body["per_attacker"]


def test_rejects_non_image(client):
    r = client.post("/cloak", files={"file": ("a.txt", b"hi", "text/plain")})
    assert r.status_code == 415


def test_rejects_too_many_steps(client):
    r = client.post("/cloak", files={"file": ("a.png", _png(), "image/png")}, data={"steps": "10000"})
    assert r.status_code == 422


def test_rejects_corrupt_image(client):
    r = client.post("/cloak", files={"file": ("a.png", b"not an image", "image/png")}, data={"steps": "2"})
    assert r.status_code == 422
