from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Literal
from uuid import UUID, uuid4

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field

LOGGER = logging.getLogger(__name__)
router = APIRouter(prefix="/evaluation", tags=["evaluation"])

GENERATOR_NAMES = ("SDXL", "FLUX", "DALL-E", "Firefly", "Midjourney")


class EvaluationRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    active_version: str = Field(min_length=1)
    candidate_checkpoint_uri: str = Field(min_length=1)
    evaluation_dataset_version: str = Field(min_length=1)


class GeneratorEvaluation(BaseModel):
    accuracy: float | None = Field(default=None, ge=0.0, le=1.0)
    sample_count: int = Field(default=0, ge=0)


class EvaluationBreakdown(BaseModel):
    generators: dict[str, GeneratorEvaluation]
    attacker_resistance_scores: dict[str, float] = Field(
        description="Resistance percentages from 0 (reconstructed) to 100."
    )


class EvaluationRunResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: UUID
    status: Literal["QUEUED", "RUNNING", "COMPLETED", "FAILED"]
    active_version: str
    candidate_checkpoint_uri: str
    evaluation_dataset_version: str
    created_at: datetime
    completed_at: datetime | None = None
    breakdown: EvaluationBreakdown | None = None
    error: str | None = None


async def _run_evaluation(app, run_id: UUID) -> None:
    """Execute the configured evaluator and update this process's run registry.

    Configure ``app.state.evaluation_runner`` with an async callable accepting
    ``(active_version, candidate_checkpoint_uri, evaluation_dataset_version)``
    and returning a mapping with ``generators`` and
    ``attacker_resistance_scores``.
    """
    runs: dict[UUID, EvaluationRunResponse] = app.state.evaluation_runs
    current = runs[run_id]
    runs[run_id] = current.model_copy(update={"status": "RUNNING"})

    runner = getattr(app.state, "evaluation_runner", None)
    if runner is None:
        runs[run_id] = current.model_copy(
            update={
                "status": "FAILED",
                "completed_at": datetime.now(timezone.utc),
                "error": "No evaluation_runner is configured",
            }
        )
        LOGGER.error("Evaluation run %s failed: no runner configured", run_id)
        return

    try:
        result = await runner(
            current.active_version,
            current.candidate_checkpoint_uri,
            current.evaluation_dataset_version,
        )
        breakdown = EvaluationBreakdown.model_validate(
            {
                "generators": result["generators"],
                "attacker_resistance_scores": result["attacker_resistance_scores"],
            }
        )
        for generator in GENERATOR_NAMES:
            breakdown.generators.setdefault(generator, GeneratorEvaluation())

        runs[run_id] = runs[run_id].model_copy(
            update={
                "status": "COMPLETED",
                "completed_at": datetime.now(timezone.utc),
                "breakdown": breakdown,
            }
        )
        LOGGER.info("Evaluation run %s completed", run_id)
    except Exception as exc:
        LOGGER.exception("Evaluation run %s failed", run_id)
        runs[run_id] = runs[run_id].model_copy(
            update={
                "status": "FAILED",
                "completed_at": datetime.now(timezone.utc),
                "error": str(exc),
            }
        )


@router.post(
    "/run",
    response_model=EvaluationRunResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def start_evaluation(
    payload: EvaluationRunRequest,
    background_tasks: BackgroundTasks,
    request: Request,
) -> EvaluationRunResponse:
    """Queue evaluation of a candidate against the configured evaluation set."""
    if not hasattr(request.app.state, "evaluation_runs"):
        request.app.state.evaluation_runs = {}

    run = EvaluationRunResponse(
        run_id=uuid4(),
        status="QUEUED",
        active_version=payload.active_version,
        candidate_checkpoint_uri=payload.candidate_checkpoint_uri,
        evaluation_dataset_version=payload.evaluation_dataset_version,
        created_at=datetime.now(timezone.utc),
    )
    request.app.state.evaluation_runs[run.run_id] = run
    background_tasks.add_task(_run_evaluation, request.app, run.run_id)
    return run


@router.get("/{run_id}", response_model=EvaluationRunResponse)
async def get_evaluation_run(
    run_id: UUID,
    request: Request,
) -> EvaluationRunResponse:
    """Fetch evaluation status and, when complete, per-generator results."""
    runs = getattr(request.app.state, "evaluation_runs", {})
    run = runs.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Evaluation run not found")
    return run
