import logging
import os
from datetime import datetime, timezone
from hashlib import sha256
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_optional_session
from ..inference import InvalidImage, ModelUnavailable, detector
from ..models import Artwork, Detection
from ..storage import storage
from .schemas import DetectionResponse

LOGGER = logging.getLogger(__name__)
router = APIRouter(tags=["detections"])

ORIGINALS_BUCKET = "originals"
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "15"))
# Uploaded images are only kept when this is on AND object storage is configured.
STORE_UPLOADS = os.getenv("STORE_UPLOADS", "false").lower() in {"1", "true", "yes"}


@router.post("/detect", response_model=DetectionResponse)
async def detect(
    file: UploadFile = File(..., description="JPG, PNG or WebP image"),
    artist_id: str = Form(default="anonymous", min_length=1, max_length=200),
    session: AsyncSession | None = Depends(get_optional_session),
) -> DetectionResponse:
    """Estimate whether an image is AI-generated or human-made."""
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Please upload an image (JPG, PNG or WebP).")

    content = await file.read(MAX_UPLOAD_MB * 1024 * 1024 + 1)
    if not content:
        raise HTTPException(status_code=422, detail="The uploaded file is empty.")
    if len(content) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"The image is larger than {MAX_UPLOAD_MB} MB.")

    try:
        result = await run_in_threadpool(detector.analyze, content)
    except InvalidImage as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ModelUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    now = datetime.now(timezone.utc)
    detection_id = None
    if session is not None:
        try:
            detection_id = await _save(session, content, file, artist_id, result)
        except Exception:  # history is a nice-to-have; never fail the answer because of it
            LOGGER.exception("Could not save detection")
            await session.rollback()

    return DetectionResponse(id=detection_id, timestamp=now, **result)


async def _save(session: AsyncSession, content: bytes, file: UploadFile, artist_id: str, result: dict):
    checksum = sha256(content).hexdigest()
    artwork = await session.scalar(select(Artwork).where(Artwork.sha256 == checksum))
    if artwork is None:
        uri = f"sha256:{checksum}"  # image itself not kept
        if STORE_UPLOADS and storage.enabled:
            key = f"{uuid4().hex}/{os.path.basename(file.filename or 'upload')}"
            await storage.put_object(
                bucket=ORIGINALS_BUCKET, key=key, body=content, content_type=file.content_type
            )
            uri = f"s3://{ORIGINALS_BUCKET}/{key}"
        artwork = Artwork(artist_id=artist_id, original_uri=uri, sha256=checksum)
        session.add(artwork)
        await session.flush()

    detection = Detection(
        artwork_id=artwork.id,
        model_version=result["model_version"],
        prediction=result["prediction"],
        ai_probability=result["ai_probability"],
        confidence=result["confidence"],
        generator=result["generator"] or "-",
    )
    session.add(detection)
    await session.commit()
    return detection.id
