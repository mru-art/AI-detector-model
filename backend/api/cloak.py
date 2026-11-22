"""Glaze-style cloaking endpoint.

    POST /cloak   (multipart: file, epsilon=8, steps=<CLOAK_STEPS>, robust=true)

Returns the cloaked PNG as base64 together with its SSIM against the original and its
health score (see ml/masking/health.py). Nothing is stored on the server.

The encoder is SCRAM's own checkpoint, found the same way as the detector (MODEL_PATH, or the
newest saved model). Cloaking runs on the CPU and takes a while, so one job runs at a time.
"""
from __future__ import annotations

import base64
import io
import os
import threading
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from ..inference import InvalidImage, ModelUnavailable, MAX_PIXELS, find_checkpoint
from .detections import MAX_UPLOAD_MB

router = APIRouter(prefix="/cloak", tags=["cloaking"])

CLOAK_STEPS = int(os.getenv("CLOAK_STEPS", "100"))
CLOAK_MAX_STEPS = int(os.getenv("CLOAK_MAX_STEPS", "300"))

_lock = threading.Lock()               # one cloaking job at a time
_encoder_lock = threading.Lock()
_encoder: Any = None
_encoder_path: str | None = None


class CloakResponse(BaseModel):
    image_png_base64: str
    ssim: float
    health: float
    per_attacker: dict[str, float]
    weakest_attack: str
    weakest_score: float
    epsilon_out_of_255: float
    steps: int
    robust: bool


def _load_encoder():
    """Load the SCRAM checkpoint once, the first time it is needed."""
    global _encoder, _encoder_path
    with _encoder_lock:
        path = find_checkpoint()
        if path is None:
            raise ModelUnavailable("No SCRAM checkpoint found. Set MODEL_PATH to a .pth file.")
        if _encoder is None or _encoder_path != str(path):
            from ml.masking import load_scram_encoder

            _encoder = load_scram_encoder(path, device="cpu")
            _encoder_path = str(path)
        return _encoder


def _cloak_job(content: bytes, epsilon: float, steps: int, robust: bool) -> dict[str, Any]:
    from PIL import Image

    from ml.masking import CloakConfig, cloak, image_ssim, protection_health

    try:
        original = Image.open(io.BytesIO(content))
        original.load()
    except Exception as exc:
        raise InvalidImage("The upload is not a readable image.") from exc
    original = original.convert("RGB")
    if original.width * original.height > MAX_PIXELS:
        raise InvalidImage("The image is too large to cloak.")

    encoder = _load_encoder()
    config = CloakConfig(epsilon=epsilon / 255.0, steps=steps, robust=robust)
    with _lock:
        cloaked = cloak(original, encoder, config, device="cpu")
        report = protection_health(original, cloaked, encoder, device="cpu")
    buf = io.BytesIO()
    cloaked.save(buf, format="PNG")
    worst_name, worst_value = report.worst_attacker if report.per_attacker else ("n/a", 0.0)
    return {
        "image_png_base64": base64.b64encode(buf.getvalue()).decode("ascii"),
        "ssim": round(image_ssim(original, cloaked), 4),
        "health": report.score,
        "per_attacker": report.per_attacker,
        "weakest_attack": worst_name,
        "weakest_score": worst_value,
        "epsilon_out_of_255": epsilon,
        "steps": steps,
        "robust": robust,
    }


@router.post("", response_model=CloakResponse)
async def cloak_artwork(
    file: UploadFile = File(..., description="Artwork to protect (PNG, JPG or WebP)"),
    epsilon: float = Form(default=8.0, ge=1.0, le=16.0, description="Max change per pixel, out of 255"),
    steps: int = Form(default=CLOAK_STEPS, ge=1, description="Optimization steps"),
    robust: bool = Form(default=True, description="Optimize through resizing and noise"),
) -> CloakResponse:
    """Cloak an artwork and return it with its SSIM and health score."""
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Please upload an image (PNG, JPG or WebP).")
    limit = MAX_UPLOAD_MB * 1024 * 1024
    content = await file.read(limit + 1)
    if not content:
        raise HTTPException(status_code=422, detail="The uploaded file is empty.")
    if len(content) > limit:
        raise HTTPException(status_code=413, detail=f"The image is larger than {MAX_UPLOAD_MB} MB.")
    if steps > CLOAK_MAX_STEPS:
        raise HTTPException(status_code=422, detail=f"steps must be at most {CLOAK_MAX_STEPS} on this server.")

    try:
        result = await run_in_threadpool(_cloak_job, content, epsilon, steps, robust)
    except InvalidImage as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ModelUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:  # the SSIM floor was not met
        raise HTTPException(status_code=422, detail=f"{exc}. Lower epsilon and try again.") from exc
    return CloakResponse(**result)
