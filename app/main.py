import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app import database as db
from app.api import chat, documents
from app.config import get_settings
from app.schemas import HealthResponse
from app.security import require_key
from app.services.vector_store import get_vector_store

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("docucite")


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    s.ensure_dirs()
    db.init_db()
    store = get_vector_store()
    stats = db.count_stats()
    if store.count != stats["chunks"]:
        log.warning("Index (%s vectors) and DB (%s chunks) differ. Call POST /api/v1/admin/reindex.",
                    store.count, stats["chunks"])
    if not s.access_key:
        log.warning("ACCESS_KEY is empty: fine on localhost, but set one before exposing this server.")
    yield


# Docs routes are defined manually below so they can be switched off with ENABLE_DOCS=false.
app = FastAPI(
    title="DocuCite RAG",
    description="Document Q&A that answers only from your files, with strict citations.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url=None, redoc_url=None, openapi_url=None,
)

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.include_router(documents.router, dependencies=[Depends(require_key)])
app.include_router(chat.router, dependencies=[Depends(require_key)])


def _docs_or_404() -> None:
    if not get_settings().enable_docs:
        raise HTTPException(404, "Not found.")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return FileResponse(STATIC_DIR / "favicon.svg", media_type="image/svg+xml")


@app.get("/openapi.json", include_in_schema=False)
def openapi_json():
    _docs_or_404()
    return JSONResponse(app.openapi())


@app.get("/docs", include_in_schema=False)
def swagger_ui():
    _docs_or_404()
    return get_swagger_ui_html(openapi_url="/openapi.json", title="DocuCite RAG · API",
                               swagger_favicon_url="/static/favicon.svg")


@app.get("/api/v1/health", response_model=HealthResponse, tags=["system"])
def health():
    stats = db.count_stats()
    vectors = get_vector_store().count
    ok = vectors == stats["chunks"]
    return HealthResponse(status="ok" if ok else "index_out_of_sync", vectors=vectors, **stats)


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    log.exception("Unhandled error on %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})
