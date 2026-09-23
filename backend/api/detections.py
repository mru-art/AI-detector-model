from hashlib import sha256
from uuid import UUID, uuid4

from botocore.exceptions import BotoCoreError, ClientError
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_session
from ..models import Artwork, Detection
from ..schemas import DetectionResponse
from ..storage import storage

router = APIRouter(tags=["detections"])
ORIGINALS_BUCKET = "originals"
MODEL_VERSION = "v1.0"


@router.post("/detect", response_model=DetectionResponse)
async def detect(
    artwork_id: UUID | None = Form(default=None),
    file: UploadFile | None = File(default=None),
    artist_id: str = Form(default="unknown", min_length=1),
    session: AsyncSession = Depends(get_session),
) -> DetectionResponse:
    if (artwork_id is None) == (file is None):
        raise HTTPException(
            status_code=422,
            detail="Provide exactly one of artwork_id or file",
        )

    if artwork_id is not None:
        artwork = await session.get(Artwork, artwork_id)
        if artwork is None:
            raise HTTPException(status_code=404, detail="Artwork not found")
    else:
        assert file is not None
        if not file.filename:
            raise HTTPException(status_code=422, detail="An image file is required")

        content = await file.read()
        if not content:
            raise HTTPException(status_code=422, detail="Uploaded file is empty")

        key = f"{uuid4().hex}/{file.filename}"
        try:
            await storage.put_object(
                bucket=ORIGINALS_BUCKET,
                key=key,
                body=content,
                content_type=file.content_type,
            )
        except (BotoCoreError, ClientError) as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Object storage is unavailable",
            ) from exc

        artwork = Artwork(
            artist_id=artist_id,
            original_uri=f"s3://{ORIGINALS_BUCKET}/{key}",
            sha256=sha256(content).hexdigest(),
        )
        session.add(artwork)
        await session.flush()

    # Replace this stub with the inference service call.
    result = {
        "ai_probability": 0.874,
        "confidence": 0.92,
        "generator": "SDXL",
    }
    detection = Detection(
        artwork_id=artwork.id,
        model_version=MODEL_VERSION,
        prediction="AI" if result["ai_probability"] >= 0.5 else "REAL",
        **result,
    )
    session.add(detection)
    await session.commit()
    await session.refresh(detection)

    return DetectionResponse.model_validate(detection)
