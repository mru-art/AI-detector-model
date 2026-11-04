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


class SourceProbability(BaseModel):
    id: str
    label: str
    probability: float = Field(ge=0.0, le=1.0)
    is_real: bool


class Region(BaseModel):
    """A 224px crop of the original image, as fractions (0-1) of the image size."""

    x: float
    y: float
    w: float
    h: float
    ai_probability: float = Field(ge=0.0, le=1.0)


class TileMap(BaseModel):
    rows: int
    cols: int
    values: list[list[float]]
    regions: list[Region]


class CropAgreement(BaseModel):
    agree: int
    total: int


class ImageSize(BaseModel):
    width: int
    height: int


class DetectionResponse(BaseModel):
    id: UUID | None = Field(default=None, description="Saved detection id (only with a database)")
    model_version: str
    prediction: str = Field(description='"AI" or "REAL"')
    ai_probability: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0, description="Probability of the verdict that was given")
    generator: str | None = Field(description="Most likely generator when the verdict is AI")
    sources: list[SourceProbability]
    uncertainty: float = Field(description="Normalised entropy: 0 = sure, 1 = clueless")
    crops: CropAgreement
    tiles: TileMap | None
    notes: list[str]
    image: ImageSize
    elapsed_ms: int
    timestamp: datetime


class ModelClass(BaseModel):
    id: str
    label: str


class ModelStatus(BaseModel):
    loaded: bool
    loading: bool
    version: str | None
    task: str | None
    classes: list[ModelClass]
    device: str | None
    macro_f1: float | None
    error: str | None


class HealthResponse(BaseModel):
    status: str
    model: ModelStatus
    database: bool
    storage: bool
    max_upload_mb: int
