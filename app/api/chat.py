from fastapi import APIRouter, HTTPException

from app.schemas import AskRequest, AskResponse
from app.services.rag_engine import LLMError, LLMRateLimitError, answer_question

router = APIRouter(prefix="/api/v1", tags=["chat"])


@router.post("/chat/ask", response_model=AskResponse)
def ask(req: AskRequest):
    try:
        return answer_question(req.question.strip(), req.top_k)
    except LLMRateLimitError as e:
        raise HTTPException(429, str(e))
    except LLMError as e:
        raise HTTPException(502, str(e))
