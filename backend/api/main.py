from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from botocore.exceptions import BotoCoreError, ClientError
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .artworks import router as artworks_router
from .detections import router as detections_router
from ..database import dispose_database
from ..storage import storage


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    await storage.ensure_buckets()
    try:
        yield
    finally:
        await dispose_database()


app = FastAPI(title="AI Art Forensics API", lifespan=lifespan)
app.include_router(artworks_router)
app.include_router(detections_router)


@app.exception_handler(BotoCoreError)
@app.exception_handler(ClientError)
async def object_storage_error_handler(
    _: Request,
    exc: BotoCoreError | ClientError,
) -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={"detail": "Object storage is unavailable"},
    )


from .continual_learning import router as continual_learning_router
from .evaluation import router as evaluation_router

app.include_router(continual_learning_router)
app.include_router(evaluation_router)
