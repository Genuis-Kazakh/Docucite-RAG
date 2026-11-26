import os
import tempfile

# Must be set BEFORE app modules are imported.
os.environ.update(DATA_DIR=tempfile.mkdtemp(prefix="docucite_test_"), EMBEDDING_BACKEND="hash",
                  LLM_API_KEY="", MIN_SCORE="0.25")

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c
        for d in c.get("/api/v1/documents").json():
            c.delete(f"/api/v1/documents/{d['id']}")
