from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ArtworkResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    artist_id: str
    original_uri: str
    protected_uri: str | None
    created_at: datetime
    sha256: str


class ArtworkDetailResponse(ArtworkResponse):
    original_download_url: str
    protected_download_url: str | None


class DetectionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    artwork_id: UUID
    model_version: str
    prediction: str
    ai_probability: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    generator: str
    timestamp: datetime
