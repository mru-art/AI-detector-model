from datetime import datetime, timezone
from uuid import UUID, uuid4

from sqlalchemy import DateTime, Float, ForeignKey, Index, Integer, String
from sqlalchemy.dialects.postgresql import UUID as PostgreSQLUUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Artwork(Base):
    __tablename__ = "artworks"
    __table_args__ = (
        Index("ix_artworks_artist_id", "artist_id"),
        Index("ix_artworks_sha256", "sha256", unique=True),
    )

    id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True), primary_key=True, default=uuid4
    )
    artist_id: Mapped[str] = mapped_column(String, nullable=False)
    original_uri: Mapped[str] = mapped_column(String, nullable=False)
    protected_uri: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    sha256: Mapped[str] = mapped_column(String, nullable=False, unique=True)

    detections: Mapped[list["Detection"]] = relationship(
        back_populates="artwork",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class Detection(Base):
    __tablename__ = "detections"
    __table_args__ = (
        Index("ix_detections_artwork_id", "artwork_id"),
        Index("ix_detections_model_version", "model_version"),
    )

    id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True), primary_key=True, default=uuid4
    )
    artwork_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("artworks.id", ondelete="CASCADE"),
        nullable=False,
    )
    model_version: Mapped[str] = mapped_column(String, nullable=False)
    prediction: Mapped[str] = mapped_column(String, nullable=False)
    ai_probability: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    generator: Mapped[str] = mapped_column(String, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )

    artwork: Mapped[Artwork] = relationship(back_populates="detections")


class ModelVersion(Base):
    __tablename__ = "model_versions"

    version: Mapped[str] = mapped_column(String, primary_key=True)
    parent_version: Mapped[str | None] = mapped_column(String, nullable=True)
    training_experience: Mapped[int] = mapped_column(Integer, nullable=False)
    dataset_version: Mapped[str] = mapped_column(String, nullable=False)
    checkpoint_uri: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
