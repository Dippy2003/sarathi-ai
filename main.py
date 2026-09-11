"""FastAPI app exposing the router -> specialist -> composer pipeline.

Usage:
    uvicorn main:app --reload
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from pipeline import answer_query

app = FastAPI(title="Sarathi AI")


class AskRequest(BaseModel):
    query: str


class AskResponse(BaseModel):
    domain: str
    needs_clarification: bool
    clarifying_question: str | None = None
    english_answer: str | None = None
    sinhala_answer: str | None = None
    sources: list[str] | None = None


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query must not be empty")

    result = answer_query(query)

    return AskResponse(
        domain=result.domain,
        needs_clarification=result.needs_clarification,
        clarifying_question=result.clarifying_question,
        english_answer=result.english_answer,
        sinhala_answer=result.sinhala_answer,
        sources=result.sources,
    )
