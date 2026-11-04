"""Kept for older instructions. The API now runs directly:

    uvicorn backend.api.main:app --reload --env-file local_dev/.env
"""
from backend.api.main import app  # noqa: F401
