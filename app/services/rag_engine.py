"""Retrieval (vector + keyword hybrid) and citation-enforced answer generation."""
import logging
import re
import time

import httpx

from app import database as db
from app.config import get_settings
from app.schemas import AskResponse, Citation
from app.services.vector_store import get_vector_store, tokenize

log = logging.getLogger("docucite.rag")

NO_ANSWER_TOKEN = "NO_ANSWER"
NOT_FOUND_TEXT = "I could not find the answer to this question in the uploaded documents."

SYSTEM_PROMPT = f"""You are a document-grounded assistant.
Rules:
1. Use ONLY the numbered context passages provided. Never use outside knowledge.
2. After every fact, add the passage number in square brackets, e.g. [1] or [2][3].
3. Never invent facts, numbers, names or sources.
4. If the context does not contain the answer, reply with exactly: {NO_ANSWER_TOKEN}
5. Answer in the same language as the question. Be concise."""


class LLMError(Exception):
    pass


class LLMRateLimitError(LLMError):
    """Provider quota exhausted (HTTP 429). Retrying immediately will not help."""


def _label(filename: str, page, section) -> str:
    if page is not None:
        return f"{filename}, p. {page}"
    if section:
        return f'{filename}, section "{section}"'
    return filename


def retrieve(question: str, k: int) -> list[dict]:
    """Vector search, re-ranked with keyword overlap (hybrid), filtered by min_score."""
    s = get_settings()
    candidates = get_vector_store().search(question, max(k * 3, 10))
    if not candidates:
        return []
    vec_score = dict(candidates)
    rows = db.get_chunks_by_ids([cid for cid, _ in candidates])
    q_tokens = set(tokenize(question))
    hits = []
    for r in rows:
        overlap = len(q_tokens & set(tokenize(r["text"]))) / len(q_tokens) if q_tokens else 0.0
        r["score"] = round(0.7 * vec_score[r["id"]] + 0.3 * overlap, 4)
        hits.append(r)
    hits = [h for h in hits if h["score"] >= s.min_score]
    hits.sort(key=lambda h: h["score"], reverse=True)
    return hits[:k]


def build_context(hits: list[dict]) -> str:
    return "\n\n".join(
        f"[{i}] ({_label(h['filename'], h['page'], h['section'])})\n{h['text']}"
        for i, h in enumerate(hits, start=1)
    )


def _call_model(model: str, question: str, context: str) -> str:
    s = get_settings()
    url = f"{s.llm_base_url.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {s.llm_api_key}"}
    body = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
    }
    retry_status = {500, 502, 503, 504}   # transient server errors only
    max_attempts = 4

    for attempt in range(1, max_attempts + 1):
        try:
            resp = httpx.post(url, headers=headers, json=body, timeout=s.llm_timeout)
            if resp.status_code == 429:
                raise LLMRateLimitError(
                    "LLM rate limit reached (provider quota). Please wait and try again later.")
            if resp.status_code in retry_status and attempt < max_attempts:
                log.warning("LLM returned %s, retry %s/%s", resp.status_code, attempt, max_attempts - 1)
                time.sleep(2 ** (attempt - 1))  # 1s, 2s, 4s
                continue
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except LLMRateLimitError:
            raise
        except httpx.TransportError as e:  # network / timeout
            if attempt < max_attempts:
                time.sleep(2 ** (attempt - 1))
                continue
            raise LLMError(f"LLM request failed: {e}") from e
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as e:
            raise LLMError(f"LLM request failed: {e}") from e
    raise LLMError("LLM request failed after retries.")


def call_llm(question: str, context: str) -> str:
    """Try the main model, then each fallback model (every model has its own quota)."""
    s = get_settings()
    models = [s.llm_model] + [m.strip() for m in s.llm_fallback_models.split(",") if m.strip()]
    last: LLMError | None = None
    for i, model in enumerate(models):
        try:
            return _call_model(model, question, context)
        except LLMError as e:
            last = e
            if i + 1 < len(models):
                log.warning("Model %s failed (%s). Trying %s.", model, e, models[i + 1])
    raise last


def _citation(i: int, h: dict) -> Citation:
    snippet = h["text"] if len(h["text"]) <= 300 else h["text"][:300].rstrip() + "…"
    return Citation(source_id=i, label=_label(h["filename"], h["page"], h["section"]),
                    filename=h["filename"], page=h["page"], section=h["section"],
                    snippet=snippet, score=h["score"],
                    document_id=h["document_id"], chunk_id=h["id"])


def answer_question(question: str, top_k: int | None = None) -> AskResponse:
    s = get_settings()
    hits = retrieve(question, top_k or s.top_k)
    if not hits:
        return AskResponse(answer=NOT_FOUND_TEXT, answerable=False, citations=[])

    all_citations = [_citation(i, h) for i, h in enumerate(hits, start=1)]

    if not s.llm_api_key:  # extractive fallback: best passage, still cited
        best = hits[0]
        text = f"{best['text']} [1]"
        return AskResponse(answer=text, answerable=True, mode="extractive",
                           citations=all_citations[:1])

    reply = call_llm(question, build_context(hits))
    if NO_ANSWER_TOKEN in reply.upper().replace(" ", "_")[:40]:
        return AskResponse(answer=NOT_FOUND_TEXT, answerable=False, citations=[])

    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", reply) if 1 <= int(n) <= len(hits)})
    if not cited:
        log.warning("LLM answer had no valid citations; returning all retrieved passages.")
        citations = all_citations
    else:
        citations = [c for c in all_citations if c.source_id in cited]
    return AskResponse(answer=reply, answerable=True, mode="llm", citations=citations)
