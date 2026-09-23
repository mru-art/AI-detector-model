from hashlib import sha256
from uuid import uuid4

from botocore.exceptions import BotoCoreError, ClientError
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_session
from ..models import Artwork
from ..schemas import ArtworkDetailResponse, ArtworkResponse
from ..storage import storage

router = APIRouter(prefix="/artworks", tags=["artworks"])
ORIGINALS_BUCKET = "originals"


def _object_key(uri: str) -> str:
    """Accept stored S3 URIs as s3://bucket/key or plain object keys."""
    if uri.startswith("s3://"):
        _, _, remainder = uri.partition("s3://")
        _, _, key = remainder.partition("/")
        return key
    return uri


@router.post("", response_model=ArtworkResponse, status_code=status.HTTP_201_CREATED)
async def create_artwork(
    artist_id: str = Form(min_length=1),
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_session),
) -> ArtworkResponse:
    if not file.filename:
        raise HTTPException(status_code=422, detail="An image file is required")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=422, detail="Uploaded file is empty")

    checksum = sha256(content).hexdigest()
    existing = await session.scalar(select(Artwork).where(Artwork.sha256 == checksum))
    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An artwork with this file checksum already exists",
        )

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
        sha256=checksum,
    )
    session.add(artwork)

    try:
        await session.commit()
    except IntegrityError as exc:
        await session.rollback()
        # Covers a concurrent request inserting the same checksum.
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An artwork with this file checksum already exists",
        ) from exc

    await session.refresh(artwork)
    return ArtworkResponse.model_validate(artwork)


@router.get("/{artwork_id}", response_model=ArtworkDetailResponse)
async def get_artwork(
    artwork_id: str,
    session: AsyncSession = Depends(get_session),
) -> ArtworkDetailResponse:
    artwork = await session.get(Artwork, artwork_id)
    if artwork is None:
        raise HTTPException(status_code=404, detail="Artwork not found")

    try:
        async with storage.client() as s3:
            original_url = await s3.generate_presigned_url(
                "get_object",
                Params={
                    "Bucket": ORIGINALS_BUCKET,
                    "Key": _object_key(artwork.original_uri),
                },
                ExpiresIn=3600,
            )

            protected_url: str | None = None
            if artwork.protected_uri:
                protected_bucket = "protected"
                protected_url = await s3.generate_presigned_url(
                    "get_object",
                    Params={
                        "Bucket": protected_bucket,
                        "Key": _object_key(artwork.protected_uri),
                    },
                    ExpiresIn=3600,
                )
    except (BotoCoreError, ClientError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Object storage is unavailable",
        ) from exc

    return ArtworkDetailResponse(
        **ArtworkResponse.model_validate(artwork).model_dump(),
        original_download_url=original_url,
        protected_download_url=protected_url,
    )
