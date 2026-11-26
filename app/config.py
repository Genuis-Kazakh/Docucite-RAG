"""Application settings loaded from environment / .env (Pydantic BaseSettings)."""
from functools import lru_cache
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    data_dir: Path = Path("data")

    chunk_size: int = Field(500, gt=50)
    chunk_overlap: int = Field(80, ge=0)

    top_k: int = Field(4, ge=1, le=20)
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embedding_backend: str = "sentence-transformers"  # or "hash" (offline/tests)
    min_score: float = Field(0.25, ge=0.0, le=1.0)    # below this -> "no answer"
    llm_timeout: float = 60.0

    # Security (both optional; defaults suit local use on 127.0.0.1)
    access_key: str = ""      # if set, every /api/v1 call needs header X-API-Key
    enable_docs: bool = True  # False hides /docs and /openapi.json

    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    llm_fallback_models: str = ""   # comma-separated backups, tried in order

    max_upload_mb: int = 20
    allowed_extensions: tuple[str, ...] = (".pdf", ".docx", ".md", ".txt")

    @model_validator(mode="after")
    def _check_overlap(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self

    @property
    def upload_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"

    @property
    def db_path(self) -> Path:
        return self.index_dir / "docucite.db"

    @property
    def faiss_path(self) -> Path:
        return self.index_dir / "faiss.index"

    def ensure_dirs(self) -> None:
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()
