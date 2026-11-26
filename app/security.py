"""Optional shared-secret protection for the API (enabled by setting ACCESS_KEY in .env)."""
import secrets

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

from app.config import get_settings

_header = APIKeyHeader(name="X-API-Key", auto_error=False,
                       description="Required only when ACCESS_KEY is set in .env")


def require_key(key: str | None = Security(_header)) -> None:
    expected = get_settings().access_key
    if expected and not secrets.compare_digest((key or "").encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="Access key required.")
