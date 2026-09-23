from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select

from ..database import AsyncSessionFactory
from ..models import ModelVersion

router = APIRouter(prefix="/continual-learning", tags=["continual-learning"])


class CheckpointMetadata(BaseModel):
    version: str
    parent_version: str | None


class ContinualMetrics(BaseModel):
    stability: float = Field(ge=0.0)
    plasticity: float = Field(ge=0.0)
    generalization: float = Field(ge=0.0)
    forgetting_rate: float
    calibration_ece: float = Field(ge=0.0)


class ContinualLearningHealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    checkpoint: CheckpointMetadata | None
    metrics: ContinualMetrics
    catastrophic_forgetting_status: Literal["LOW", "MEDIUM", "HIGH"]


class ModelUpdateHistoryItem(BaseModel):
    version: str
    accuracy: float = Field(ge=0.0, le=1.0)
    stability: float = Field(ge=0.0)


class ModelUpdateHistoryResponse(BaseModel):
    updates: list[ModelUpdateHistoryItem]


def _status(forgetting_rate: float) -> Literal["LOW", "MEDIUM", "HIGH"]:
    if forgetting_rate <= 0.05:
        return "LOW"
    if forgetting_rate <= 0.15:
        return "MEDIUM"
    return "HIGH"


@router.get("/health", response_model=ContinualLearningHealthResponse)
async def get_continual_learning_health(
    request: Request,
) -> ContinualLearningHealthResponse:
    """Return the newest checkpoint and metrics registered by the trainer.

    The trainer should set ``app.state.continual_metrics`` to a dictionary
    containing the fields in ``ContinualMetrics`` after each accepted update.
    """
    async with AsyncSessionFactory() as session:
        latest = await session.scalar(
            select(ModelVersion).order_by(ModelVersion.created_at.desc()).limit(1)
        )

    raw_metrics = getattr(request.app.state, "continual_metrics", None)
    if raw_metrics is None:
        raise HTTPException(
            status_code=503,
            detail="Continual-learning metrics have not been registered",
        )

    metrics = ContinualMetrics.model_validate(raw_metrics)
    return ContinualLearningHealthResponse(
        checkpoint=(
            CheckpointMetadata(
                version=latest.version,
                parent_version=latest.parent_version,
            )
            if latest
            else None
        ),
        metrics=metrics,
        catastrophic_forgetting_status=_status(metrics.forgetting_rate),
    )


@router.get("/metrics/history", response_model=ModelUpdateHistoryResponse)
async def get_metrics_history(
    request: Request,
) -> ModelUpdateHistoryResponse:
    """Return update history recorded by the continual-learning trainer."""
    history = getattr(request.app.state, "continual_metrics_history", None)
    if history is None:
        raise HTTPException(
            status_code=503,
            detail="Continual-learning metric history has not been registered",
        )
    return ModelUpdateHistoryResponse.model_validate({"updates": history})
