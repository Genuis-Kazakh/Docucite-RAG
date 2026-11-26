"""Evaluation pipeline: Retrieval Recall, Answer Faithfulness, Abstention accuracy.

Run:  python -m tests.eval_rag            (uses your .env; real LLM if LLM_API_KEY is set)
Offline (no LLM quota used, ignores the key in .env):
      python -m tests.eval_rag --offline
Uses an isolated temp data dir, so your real index is untouched.
EVAL_DELAY = seconds to wait between LLM calls (default 13 -> stays under 5 requests/min).
"""
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).parent
DATA = HERE / "eval_data"


def main() -> int:
    os.environ["DATA_DIR"] = tempfile.mkdtemp(prefix="docucite_eval_")
    if "--offline" in sys.argv:
        os.environ["LLM_API_KEY"] = ""   # overrides .env (PowerShell cannot set an empty env var)
    from app import database as db
    from app.config import get_settings
    from app.services.ingestion import ingest_file
    from app.services.rag_engine import LLMError, LLMRateLimitError, answer_question, retrieve

    s = get_settings()
    use_llm = bool(s.llm_api_key)
    delay = int(os.getenv("EVAL_DELAY", "13")) if use_llm else 0

    def ask(question):
        resp = answer_question(question)
        if delay:
            time.sleep(delay)
        return resp

    db.init_db()
    for f in sorted(DATA.glob("*.md")):
        ingest_file(f, f.name)

    questions = json.loads((DATA / "questions.json").read_text(encoding="utf-8"))
    ans_q = [q for q in questions if q["answerable"]]
    no_q = [q for q in questions if not q["answerable"]]
    recall = faithful = abstain = 0

    print(f"Mode: {'LLM' if use_llm else 'extractive'} | backend: {s.embedding_backend} | top_k={s.top_k}\n")
    try:
        for q in ans_q:
            hits = retrieve(q["question"], s.top_k)
            r_ok = any(h["filename"] == q["file"] and q["keyword"].lower() in h["text"].lower() for h in hits)
            resp = ask(q["question"])
            f_ok = (resp.answerable and q["keyword"].lower() in resp.answer.lower()
                    and any(c.filename == q["file"] for c in resp.citations))
            recall += r_ok
            faithful += f_ok
            print(f"[{'PASS' if r_ok and f_ok else 'FAIL'}] recall={int(r_ok)} faithful={int(f_ok)} | {q['question']}")
        for q in no_q:
            resp = ask(q["question"])
            ok = not resp.answerable
            abstain += ok
            print(f"[{'PASS' if ok else 'FAIL'}] abstain={int(ok)} | {q['question']}")
    except LLMRateLimitError:
        print("\nSTOPPED: LLM rate limit reached. Wait for the quota to reset, or run offline with LLM_API_KEY=\"\".")
        return 2
    except LLMError as e:
        print(f"\nSTOPPED: LLM error: {e}")
        return 2
    finally:
        shutil.rmtree(os.environ["DATA_DIR"], ignore_errors=True)

    print("\n===== RESULTS =====")
    print(f"Retrieval Recall@{s.top_k}:   {recall}/{len(ans_q)} = {recall/len(ans_q):.0%}")
    print(f"Answer Faithfulness:    {faithful}/{len(ans_q)} = {faithful/len(ans_q):.0%}")
    print(f"Correct abstention:     {abstain}/{len(no_q)} = {abstain/len(no_q):.0%}")
    return 0 if (recall == len(ans_q) and abstain == len(no_q)) else 1


if __name__ == "__main__":
    sys.exit(main())
