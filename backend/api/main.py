"""S.C.R.A.M API.

Run from the repo root:
    uvicorn backend.api.main:app --reload

Everything except the model is optional:
    MODEL_PATH      checkpoint to use (default: newest one under vision_transformer/)
    DATABASE_URL    save detection history (sqlite+aiosqlite:///./scram.db works)
    S3_ENDPOINT_URL object storage for artworks / protection (MinIO, S3 ...)
    CORS_ORIGINS    comma-separated origins allowed to call the API (default: *)
"""
import asyncio
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ..database import create_tables, database_enabled, dispose_database
from ..inference import detector
from ..metrics import load_metrics
from ..storage import storage
from .detections import MAX_UPLOAD_MB
from .detections import router as detections_router
from .evaluation import router as evaluation_router
from .schemas import HealthResponse

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
LOGGER = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    await create_tables()
    if storage.enabled:
        try:
            await storage.ensure_buckets()
        except Exception:
            LOGGER.exception("Object storage is configured but unreachable")
    # Load the model in the background so the server answers /health straight away.
    loader = asyncio.create_task(asyncio.to_thread(detector.load))
    try:
        yield
    finally:
        loader.cancel()
        await dispose_database()


app = FastAPI(
    title="S.C.R.A.M API",
    description="Estimates whether an artwork is AI-generated or human-made.",
    version="0.2.0",
    lifespan=lifespan,
)

origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(detections_router)
app.include_router(evaluation_router)
if database_enabled() and storage.enabled:
    from .artworks import router as artworks_router

    app.include_router(artworks_router)


@app.get("/", include_in_schema=False)
async def root() -> dict[str, str]:
    return {"name": "S.C.R.A.M API", "docs": "/docs", "health": "/health"}


@app.get("/health", response_model=HealthResponse, tags=["status"])
async def health() -> HealthResponse:
    info = detector.info.public()
    return HealthResponse(
        status="ok" if info["loaded"] else ("loading" if info["loading"] else "no-model"),
        model=info,
        database=database_enabled(),
        storage=storage.enabled,
        max_upload_mb=MAX_UPLOAD_MB,
    )


@app.get("/metrics/continual", tags=["status"])
async def continual_metrics(method: str | None = Query(default=None)) -> dict:
    """Accuracy per class after each continual-learning experience."""
    try:
        data = load_metrics(method)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"No results for method {method!r}") from exc
    if data is None:
        raise HTTPException(status_code=404, detail="No continual-learning results found.")
    return data


@app.exception_handler(Exception)
async def unexpected_error(_: Request, exc: Exception) -> JSONResponse:
    LOGGER.exception("Unhandled error", exc_info=exc)
    return JSONResponse(status_code=500, content={"detail": "Something went wrong on the server."})
