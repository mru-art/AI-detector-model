"""Local entry point that wraps the repo's FastAPI app without editing it.

Adds what the repo's backend is missing for local use:
  * CORS, so a browser on another port can call the API directly
  * table creation on startup (the repo has no migrations / create_all)
  * a clean 409 instead of a 500 when the same image is uploaded twice

Run from the repo root:
  uvicorn local_dev.run_api:app --reload --env-file local_dev/.env
"""

import os
from contextlib import asynccontextmanager

from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.exc import IntegrityError

import sys

# The repo's api/artworks.py and api/detections.py import "..schemas"
# (backend.schemas), but the file lives at backend/api/schemas.py.
# Point the wrong name at the real module instead of editing the repo.
import backend.api.schemas as _schemas

sys.modules.setdefault("backend.schemas", _schemas)

from backend.api.main import app
from backend.database import engine
from backend.models import Base

_original_lifespan = app.router.lifespan_context


@asynccontextmanager
async def _lifespan(a):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    async with _original_lifespan(a) as state:
        yield state


app.router.lifespan_context = _lifespan

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(IntegrityError)
async def _duplicate_handler(_: Request, exc: IntegrityError) -> JSONResponse:
    if "sha256" in str(exc.orig):
        return JSONResponse(
            status_code=409,
            content={"detail": "This image has already been uploaded."},
        )
    return JSONResponse(status_code=500, content={"detail": "Database error"})
