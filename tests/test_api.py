from app.services import rag_engine

DOC = (b"# Policy\n\n## Vacation\nEvery employee receives 24 vacation days per year.\n\n"
       b"## Remote\nEmployees may work remotely 2 days per week.\n")


def upload(client, name="policy.md", data=DOC):
    return client.post("/api/v1/documents/upload", files={"file": (name, data)})


def test_health_empty(client):
    r = client.get("/api/v1/health").json()
    assert r["status"] == "ok" and r["chunks"] == 0 and r["vectors"] == 0


def test_upload_list_delete_keeps_index_in_sync(client):
    r = upload(client)
    assert r.status_code == 201
    doc = r.json()["document"]
    assert doc["chunk_count"] >= 1
    assert client.get("/api/v1/health").json()["vectors"] == doc["chunk_count"]
    assert len(client.get("/api/v1/documents").json()) == 1
    assert client.delete(f"/api/v1/documents/{doc['id']}").status_code == 200
    h = client.get("/api/v1/health").json()
    assert h["vectors"] == 0 and h["chunks"] == 0 and h["status"] == "ok"
    assert client.delete(f"/api/v1/documents/{doc['id']}").status_code == 404


def test_rejects_bad_files(client):
    assert upload(client, "x.exe", b"abc").status_code == 415
    assert upload(client, "empty.txt", b"   ").status_code == 422
    assert client.post("/api/v1/documents/upload",
                       files={"file": ("bad.pdf", b"not a pdf")}).status_code == 422


def test_ask_extractive_with_citation(client):
    upload(client)
    r = client.post("/api/v1/chat/ask", json={"question": "How many vacation days per year?"}).json()
    assert r["answerable"] and "24" in r["answer"] and r["mode"] == "extractive"
    assert r["citations"][0]["filename"] == "policy.md" and r["citations"][0]["source_id"] == 1


def test_ask_unanswerable(client):
    upload(client)
    r = client.post("/api/v1/chat/ask", json={"question": "What is the capital of Australia?"}).json()
    assert r["answerable"] is False and r["citations"] == []


def test_ask_with_no_documents(client):
    r = client.post("/api/v1/chat/ask", json={"question": "Anything at all here?"}).json()
    assert r["answerable"] is False


def test_llm_mode_citations_filtered(client, monkeypatch):
    upload(client)
    monkeypatch.setenv("LLM_API_KEY", "x")
    from app.config import get_settings
    get_settings.cache_clear()
    monkeypatch.setattr(rag_engine, "call_llm", lambda q, ctx: "Employees get 24 days [1].")
    r = client.post("/api/v1/chat/ask", json={"question": "How many vacation days per year?"}).json()
    assert r["mode"] == "llm" and [c["source_id"] for c in r["citations"]] == [1]
    monkeypatch.setattr(rag_engine, "call_llm", lambda q, ctx: "NO_ANSWER")
    r = client.post("/api/v1/chat/ask", json={"question": "How many vacation days per year?"}).json()
    assert r["answerable"] is False
    monkeypatch.delenv("LLM_API_KEY")
    get_settings.cache_clear()


def test_llm_failure_returns_502(client, monkeypatch):
    upload(client)
    monkeypatch.setenv("LLM_API_KEY", "x")
    from app.config import get_settings
    get_settings.cache_clear()

    def boom(q, ctx):
        raise rag_engine.LLMError("down")
    monkeypatch.setattr(rag_engine, "call_llm", boom)
    assert client.post("/api/v1/chat/ask",
                       json={"question": "How many vacation days per year?"}).status_code == 502
    monkeypatch.delenv("LLM_API_KEY")
    get_settings.cache_clear()


def test_reindex(client):
    upload(client)
    assert client.post("/api/v1/admin/reindex").json()["reindexed_chunks"] >= 1
    assert client.get("/api/v1/health").json()["status"] == "ok"


def test_ui_is_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "DocuCite" in r.text


def test_rate_limit_maps_to_429(client, monkeypatch):
    upload(client)
    monkeypatch.setenv("LLM_API_KEY", "x")
    from app.config import get_settings
    get_settings.cache_clear()

    def limited(q, ctx):
        raise rag_engine.LLMRateLimitError("quota")
    monkeypatch.setattr(rag_engine, "call_llm", limited)
    assert client.post("/api/v1/chat/ask",
                       json={"question": "How many vacation days per year?"}).status_code == 429
    monkeypatch.delenv("LLM_API_KEY")
    get_settings.cache_clear()


def test_favicon_and_custom_docs(client):
    assert client.get("/static/favicon.svg").status_code == 200
    assert client.get("/favicon.ico").status_code == 200
    r = client.get("/docs")
    assert r.status_code == 200 and "favicon.svg" in r.text


def test_chunks_endpoint_and_citation_ids(client):
    doc = upload(client).json()["document"]
    r = client.get(f"/api/v1/documents/{doc['id']}/chunks").json()
    assert r["document"]["filename"] == "policy.md" and len(r["chunks"]) == doc["chunk_count"]
    ans = client.post("/api/v1/chat/ask", json={"question": "How many vacation days per year?"}).json()
    c = ans["citations"][0]
    assert c["document_id"] == doc["id"] and c["chunk_id"] in [x["id"] for x in r["chunks"]]
    assert client.get("/api/v1/documents/9999/chunks").status_code == 404


def test_access_key_protects_api(client, monkeypatch):
    from app.config import get_settings
    monkeypatch.setenv("ACCESS_KEY", "s3cret")
    get_settings.cache_clear()
    assert client.get("/api/v1/documents").status_code == 401
    assert client.get("/api/v1/documents", headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.post("/api/v1/chat/ask", json={"question": "anything here"}).status_code == 401
    assert client.get("/api/v1/documents", headers={"X-API-Key": "s3cret"}).status_code == 200
    assert client.get("/api/v1/health").status_code == 200
    monkeypatch.delenv("ACCESS_KEY")
    get_settings.cache_clear()


def test_docs_can_be_disabled(client, monkeypatch):
    from app.config import get_settings
    monkeypatch.setenv("ENABLE_DOCS", "false")
    get_settings.cache_clear()
    assert client.get("/docs").status_code == 404
    assert client.get("/openapi.json").status_code == 404
    assert client.get("/redoc").status_code == 404
    monkeypatch.delenv("ENABLE_DOCS")
    get_settings.cache_clear()
    assert client.get("/openapi.json").status_code == 200


def test_llm_fallback_models(monkeypatch):
    from app.config import get_settings
    monkeypatch.setenv("LLM_MODEL", "m1")
    monkeypatch.setenv("LLM_FALLBACK_MODELS", "m2, m3")
    get_settings.cache_clear()
    calls = []

    def fake(model, q, ctx):
        calls.append(model)
        if model != "m3":
            raise rag_engine.LLMRateLimitError("quota")
        return "ok [1]"
    monkeypatch.setattr(rag_engine, "_call_model", fake)
    assert rag_engine.call_llm("q", "c") == "ok [1]" and calls == ["m1", "m2", "m3"]
    monkeypatch.delenv("LLM_MODEL")
    monkeypatch.delenv("LLM_FALLBACK_MODELS")
    get_settings.cache_clear()
